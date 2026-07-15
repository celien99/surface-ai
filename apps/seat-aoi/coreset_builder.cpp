#include "coreset_builder.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "app_config.h"
#include "cli_args.h"

#include <sai/core/error.h>
#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/image/preprocess.h>
#include <sai/image/gpu_image.h>
#include <sai/inference/tensorrt_engine.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/embedding/patch_embedder.h>
#include <sai/embedding/embedding.h>
#include <sai/detection/feature_bank.h>
#include <sai/io/coreset_manifest.h>
#include <sai/memory/gpu_pool.h>
#include <sai/memory/arena_allocator.h>
#include <cuda_runtime.h>

// Build coreset from normal (defect-free) sample images.
// Uses DINOv3 → PatchEmbedder (1024-dim) with TensorRT GPU inference.
auto BuildCoreset(const CliArgs& cli) -> int {
    using namespace sai;
    using namespace seat_aoi::config;

    std::cout << "Building coreset from dataset: "
              << cli.dataset_path << "\n";

    io::BasicImporter importer;

    // ── Create DINOv3 embedder ──
    auto infer_engine = std::make_shared<inference::TensorRtEngine>(
        /*device_ordinal=*/0);
    inference::DinoV3Config dino_cfg;
    dino_cfg.engine_path = kDinoV3Engine;
    dino_cfg.image_size = kImageSize;
    dino_cfg.patch_size = kPatchSize;
    dino_cfg.embed_dim = kEmbedDim;

    auto dino_adapter = inference::DinoV3Adapter::Create(
        *infer_engine, dino_cfg);
    if (!dino_adapter) {
        std::cerr << "DinoV3Adapter creation failed: "
                  << dino_adapter.error().message << "\n";
        return 1;
    }
    auto patch_emb = embedding::PatchEmbedder::Create(
        std::move(*dino_adapter));
    if (!patch_emb) {
        std::cerr << "PatchEmbedder creation failed\n";
        return 1;
    }
    auto embedder = std::make_unique<embedding::PatchEmbedder>(
        std::move(*patch_emb));
    std::cout << "Embedder: PatchEmbedder (DINOv3, dim="
              << kEmbedDim << ")\n";

    // Read dataset manifest
    auto dataset_result = importer.ImportDataset(cli.dataset_path);
    if (!dataset_result) {
        std::cerr << "Dataset import failed: "
                  << dataset_result.error().message << "\n";
        return 1;
    }
    auto& entries = *dataset_result;
    std::cout << "Found " << entries.size() << " normal sample images\n";

    // Preprocess chain: resize to embedder's expected size
    auto preprocess_chain = image::Compose({
        image::MakeResize(kImageSize, kImageSize)
    });

    // GPU upload: SurfaceImage → GpuImage → DINOv3 → DtoH
    sai::memory::ArenaAllocator arena(64 * 1024 * 1024);  // 64 MiB metadata
    auto gpu_pool_result = sai::memory::GpuPool::Create(
        sai::memory::MemoryPoolConfig{
            .slab_size = kImageSize * kImageSize * 3,   // RGB8
            .slab_count = 16,
        },
        arena);
    if (!gpu_pool_result) {
        std::cerr << "GpuPool creation failed: "
                  << gpu_pool_result.error().message << "\n";
        return 1;
    }
    auto& gpu_pool = **gpu_pool_result;

    // Group entries by position_id for per-position coreset building
    std::map<std::uint16_t, std::vector<decltype(entries)::value_type>> pos_entries;
    std::string surface_id;
    for (auto& e : entries) {
        pos_entries[e.position_id].push_back(e);
        if (surface_id.empty()) surface_id = e.surface_id;
    }
    std::cout << "Positions: " << pos_entries.size() << "\n";

    // Group embeddings by position_id
    std::map<std::uint16_t, std::vector<embedding::Embedding>> pos_embeddings;
    int processed = 0;

    for (auto& entry : entries) {
        auto img_result = importer.ImportImage(entry.path);
        if (!img_result) {
            std::cerr << "Failed to import " << entry.entry.path.filename().string()
                      << ": " << img_result.error().message << "\n";
            continue;
        }

        // Stamp image metadata from dataset entry
        (*img_result)->Meta().surface_id = entry.surface_id;
        (*img_result)->Meta().position_id = entry.position_id;
        (*img_result)->Meta().light_id = entry.light_id;

        // Run through preprocess (resize) to get SurfaceImage
        auto preprocess_result = preprocess_chain(std::move(*img_result));
        if (!preprocess_result) {
            std::cerr << "Preprocess failed for " << entry.path.filename().string()
                      << ": " << preprocess_result.error().message << "\n";
            continue;
        }

        // Wrap raw Image* as SurfaceImage
        sai::image::SurfaceImage surface = [&]() -> sai::image::SurfaceImage {
            auto* raw = preprocess_result->get();
            if (auto* surf = dynamic_cast<image::SurfaceImage*>(raw)) {
                auto s = std::move(*surf);
                delete raw;  // release unique_ptr ownership
                return s;
            }
            auto meta = raw->Meta();
            const auto* data = raw->Data();
            auto size = raw->SizeBytes();
            std::vector<std::uint8_t> buffer(data, data + size);
            delete raw;
            return sai::image::SurfaceImage::FromOwnedBuffer(
                std::move(buffer), meta);
        }();

        // Extract embedding: HtoD → DINOv3 infer → DtoH
        sai::Result<embedding::Embedding> emb = tl::make_unexpected(
            ErrorInfo{ErrorCode::Inference_EngineExecutionFailed, "no path"});
        {
            // HtoD: upload SurfaceImage to GPU
            auto gpu_img_result = image::GpuImage::FromPool(
                gpu_pool, surface.Meta());
            if (!gpu_img_result) {
                std::cerr << "GpuImage allocation failed for "
                          << entry.path.filename().string() << "\n";
                continue;
            }
            auto gpu_img = std::move(*gpu_img_result);
            auto cuda_err = cudaMemcpy(
                gpu_img.Data(), surface.Data(), surface.SizeBytes(),
                cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                std::cerr << "cudaMemcpy HtoD failed for "
                          << entry.path.filename().string() << "\n";
                continue;
            }

            // Run DINOv3 inference on GPU
            emb = embedder->Extract(gpu_img);
            if (emb && emb->IsOnGpu()) {
                // DtoH: download patch features to CPU for FeatureBank
                auto byte_size = emb->SizeBytes();
                std::vector<float> cpu_data(emb->Meta().count * emb->Meta().dim);
                cuda_err = cudaMemcpy(
                    cpu_data.data(), emb->Data(), byte_size,
                    cudaMemcpyDeviceToHost);
                if (cuda_err != cudaSuccess) {
                    std::cerr << "cudaMemcpy DtoH failed for "
                              << entry.path.filename().string() << "\n";
                    continue;
                }
                emb = embedding::Embedding::FromCpu(
                    std::move(cpu_data), emb->Meta());
            }
        }

        if (!emb) {
            std::cerr << "Embedding extraction failed for "
                      << entry.path.filename().string() << ": "
                      << emb.error().message << "\n";
            continue;
        }

        ++processed;
        pos_embeddings[entry.position_id].push_back(std::move(*emb));
        std::cout << "[" << processed << "/" << entries.size() << "] "
                  << entry.path.filename().string() << " → "
                  << pos_embeddings[entry.position_id].back().Meta().count << " patches\n";
    }

    // Build per-position FeatureBanks and manifest
    bool use_greedy = (cli.coreset_algo == "greedy");
    std::cout << "\nCoreset algorithm: " << (use_greedy ? "greedy (furthest-point sampling)" : "uniform")
              << "\nMax samples: " << cli.coreset_max_samples << "\n\n";

    auto output_dir = std::filesystem::path(cli.coreset_output_path);
    io::CoresetManifest manifest;
    manifest.surface_id = surface_id;

    for (auto& [pid, embeddings] : pos_embeddings) {
        if (embeddings.empty()) {
            std::cerr << "Position " << pid << ": no embeddings — skipped\n";
            continue;
        }

        std::vector<const embedding::Embedding*> emb_ptrs;
        emb_ptrs.reserve(embeddings.size());
        for (auto& e : embeddings) emb_ptrs.push_back(&e);

        Result<detection::FeatureBank> bank_result = tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed, "No algorithm selected"});

        if (use_greedy) {
            bank_result = detection::FeatureBank::BuildWithGreedyCoreset(
                emb_ptrs, kEmbedDim, cli.coreset_max_samples);
        } else {
            bank_result = detection::FeatureBank::BuildFromEmbeddings(
                emb_ptrs, kEmbedDim, cli.coreset_max_samples);
        }

        if (!bank_result) {
            std::cerr << "Position " << pid << ": FeatureBank build failed: "
                      << bank_result.error().message << "\n";
            return 1;
        }

        auto bank_path = output_dir / ("pos_" + std::to_string(pid) + ".bin");
        auto save_result = bank_result->SaveToFile(bank_path);
        if (!save_result) {
            std::cerr << "Position " << pid << ": failed to save coreset: "
                      << save_result.error().message << "\n";
            return 1;
        }

        manifest.banks.push_back(io::CoresetBankEntry{
            .position_id = pid,
            .path = bank_path,
        });

        std::cout << "Position " << pid << ":\n"
                  << "  Images:       " << embeddings.size() << "\n"
                  << "  Coreset size: " << bank_result->NumSamples() << "\n"
                  << "  Feature dim:  " << bank_result->Dim() << "\n"
                  << "  Output:       " << bank_path.string() << "\n\n";
    }

    // Save manifest
    auto manifest_path = output_dir / (surface_id + ".yaml");
    auto manifest_save = io::SaveCoresetManifest(manifest_path, manifest);
    if (!manifest_save) {
        std::cerr << "Failed to save coreset manifest: "
                  << manifest_save.error().message << "\n";
        return 1;
    }

    std::cout << "Coreset manifest: " << manifest_path.string() << "\n"
              << "Total positions:  " << manifest.banks.size() << "\n"
              << "Total images:     " << processed << "\n";
    return 0;
}
