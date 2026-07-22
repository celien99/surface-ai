#include "coreset_builder.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "app_config.h"
#include "cli_args.h"
#include "embedder_factory.h"

#include <sai/core/error.h>
#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/image/preprocess.h>
#include <sai/image/gpu_image.h>
#include <sai/embedding/embedding.h>
#include <sai/detection/bounded_patch_sampler.h>
#include <sai/detection/feature_bank.h>
#include <sai/io/coreset_manifest.h>
#include <sai/memory/gpu_pool.h>
#include <sai/memory/arena_allocator.h>
#include <cuda_runtime.h>

// Build coreset from normal (defect-free) sample images.
// Uses DINOv2 → PatchEmbedder (768-dim) with TensorRT GPU inference.
auto BuildCoreset(const CliArgs& cli) -> int {
    using namespace sai;
    using namespace seat_aoi::config;

    std::cout << "Building coreset from dataset: "
              << cli.dataset_path << "\n";

    io::BasicImporter importer;

    // ── Create DINOv2 embedder ──
    auto embedder_result = seat_aoi::CreateDinoV2PatchEmbedder(
        DinoV2Engine(), kImageSize, kPatchSize, kEmbedDim);
    if (!embedder_result) {
        std::cerr << "DINOv2 embedder creation failed: "
                  << embedder_result.error().message << "\n";
        return 1;
    }
    auto embedder = std::move(*embedder_result);

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

    // GPU upload: SurfaceImage → GpuImage → DINOv2 → DtoH
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

    std::string surface_id;
    for (auto& e : entries) {
        if (surface_id.empty()) surface_id = e.surface_id;
    }

    std::map<std::uint16_t, detection::BoundedPatchSampler> pos_samplers;
    std::map<std::uint16_t, std::size_t> sampled_image_counts;
    int processed = 0;

    for (auto& entry : entries) {
        auto img_result = importer.ImportImage(entry.path);
        if (!img_result) {
            std::cerr << "Failed to import " << entry.path.filename().string()
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
            auto* raw = preprocess_result->release();  // transfer ownership from unique_ptr
            if (auto* surf = dynamic_cast<image::SurfaceImage*>(raw)) {
                auto s = std::move(*surf);
                delete raw;
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

        // Extract embedding: HtoD → DINOv2 infer → DtoH
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

            // Run DINOv2 inference on GPU
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

        auto [sampler_it, inserted] = pos_samplers.try_emplace(
            entry.position_id, kEmbedDim, cli.coreset_max_samples);
        (void)inserted;
        sampler_it->second.Add(emb->Data(), emb->Meta().count);
        ++sampled_image_counts[entry.position_id];
        ++processed;
        std::cout << "[" << processed << "/" << entries.size() << "] "
                  << entry.path.filename().string() << " → "
                  << emb->Meta().count << " patches\n";
    }

    // Build per-position FeatureBanks and manifest
    std::cout << "Positions: " << pos_samplers.size() << "\n";
    std::cout << "\nCoreset algorithm: deterministic bounded sampling"
              << "\nMax samples: " << cli.coreset_max_samples << "\n\n";

    auto output_dir = std::filesystem::path(cli.coreset_output_path);
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Failed to create output directory: " << output_dir.string() << "\n";
        return 1;
    }

    io::CoresetManifest manifest;
    manifest.surface_id = surface_id;

    for (auto& [pid, sampler] : pos_samplers) {
        if (sampler.Size() == 0) {
            std::cerr << "Position " << pid << ": no embeddings — skipped\n";
            continue;
        }

        auto bank = detection::FeatureBank::BuildFromVectors(
            sampler.Vectors().data(), sampler.Size(), kEmbedDim);

        auto bank_path = output_dir / ("pos_" + std::to_string(pid) + ".bin");
        auto save_result = bank.SaveToFile(bank_path);
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
                  << "  Images:       " << sampled_image_counts[pid] << "\n"
                  << "  Coreset size: " << bank.NumSamples() << "\n"
                  << "  Feature dim:  " << bank.Dim() << "\n"
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
