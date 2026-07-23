#include "coreset_builder.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <source_location>
#include <string>
#include <vector>

#include "app_config.h"
#include "cli_args.h"
#include "embedder_factory.h"

#include <sai/core/error.h>
#include <sai/io/importer.h>
#include <sai/image/preprocess.h>
#include <sai/image/gpu_image.h>
#include <sai/embedding/embedder.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/coreset_evolution.h>
#include <sai/io/coreset_manifest.h>
#include <sai/memory/gpu_pool.h>
#include <sai/memory/arena_allocator.h>
#include <cuda_runtime.h>

namespace {

using Clock = std::chrono::steady_clock;

struct TrainingTiming {
    Clock::duration import{};
    Clock::duration preprocess{};
    Clock::duration upload{};
    Clock::duration inference{};
    Clock::duration download{};
};

[[nodiscard]] auto AppendEmbeddingVectors(
    const sai::io::DatasetEntry& entry,
    sai::io::BasicImporter& importer,
    const sai::image::PreprocessFn& preprocess,
    sai::memory::IMemoryPool& gpu_pool,
    sai::embedding::IEmbedder& embedder,
    TrainingTiming& timing,
    std::size_t expected_dim,
    std::vector<float>& destination) -> sai::Result<std::size_t> {
    auto started = Clock::now();
    auto image_result = importer.ImportImage(entry.path);
    timing.import += Clock::now() - started;
    if (!image_result) return tl::make_unexpected(image_result.error());

    auto& meta = (*image_result)->Meta();
    meta.surface_id = entry.surface_id;
    meta.position_id = entry.position_id;
    meta.light_id = entry.light_id;

    started = Clock::now();
    auto preprocess_result = preprocess(std::move(*image_result));
    timing.preprocess += Clock::now() - started;
    if (!preprocess_result) return tl::make_unexpected(preprocess_result.error());
    auto processed = std::move(*preprocess_result);

    started = Clock::now();
    auto gpu_image_result = sai::image::GpuImage::FromPool(gpu_pool, processed->Meta());
    if (!gpu_image_result) return tl::make_unexpected(gpu_image_result.error());
    auto gpu_image = std::move(*gpu_image_result);
    auto cuda_error = cudaMemcpy(gpu_image.Data(), processed->Data(), processed->SizeBytes(),
                                 cudaMemcpyHostToDevice);
    timing.upload += Clock::now() - started;
    if (cuda_error != cudaSuccess) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            std::string("cudaMemcpy HtoD failed: ") + cudaGetErrorString(cuda_error),
            std::source_location::current(),
        });
    }

    started = Clock::now();
    auto embedding_result = embedder.Extract(gpu_image);
    timing.inference += Clock::now() - started;
    if (!embedding_result) return tl::make_unexpected(embedding_result.error());
    if (embedding_result->Meta().dim != expected_dim) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Embedding_DimensionMismatch,
            "unexpected embedding dimension",
            std::source_location::current(),
        });
    }
    const auto count = embedding_result->Meta().count;
    const auto value_count = count * embedding_result->Meta().dim;
    if (!embedding_result->IsOnGpu()) {
        destination.insert(destination.end(), embedding_result->Data(),
                           embedding_result->Data() + value_count);
        return count;
    }

    const auto old_size = destination.size();
    destination.resize(old_size + value_count);
    started = Clock::now();
    cuda_error = cudaMemcpy(destination.data() + old_size, embedding_result->Data(),
                            embedding_result->SizeBytes(), cudaMemcpyDeviceToHost);
    timing.download += Clock::now() - started;
    if (cuda_error != cudaSuccess) {
        destination.resize(old_size);
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            std::string("cudaMemcpy DtoH failed: ") + cudaGetErrorString(cuda_error),
            std::source_location::current(),
        });
    }
    return count;
}

}  // namespace

// Build coreset from normal (defect-free) sample images.
// Uses DINOv2 → PatchEmbedder (768-dim) with TensorRT GPU inference.
auto BuildCoreset(const CliArgs& cli) -> int {
    using namespace sai;
    using namespace seat_aoi::config;

    std::cout << "Building coreset from dataset: "
              << cli.dataset_path << "\n";
    const auto build_started = Clock::now();
    TrainingTiming timing;

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

    // Group entries only; embeddings are built and released one position at a time.
    std::map<std::uint16_t, std::vector<io::DatasetEntry>> pos_entries;
    std::string surface_id;
    for (auto& e : entries) {
        if (surface_id.empty()) surface_id = e.surface_id;
        pos_entries[e.position_id].push_back(std::move(e));
    }
    std::cout << "Positions: " << pos_entries.size() << "\n";

    // Build per-position FeatureBanks and manifest
    std::cout << "\nCoreset algorithm: greedy (furthest-point sampling)"
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
    std::size_t processed = 0;
    const auto total_images = entries.size();

    for (auto& [pid, position_entries] : pos_entries) {
        std::vector<float> position_vectors;
        const auto patches_per_image =
            (kImageSize / kPatchSize) * (kImageSize / kPatchSize);
        position_vectors.reserve(
            position_entries.size() * patches_per_image * kEmbedDim);
        std::size_t position_images = 0;
        for (const auto& entry : position_entries) {
            auto patch_count = AppendEmbeddingVectors(
                entry, importer, preprocess_chain, gpu_pool, *embedder, timing,
                kEmbedDim, position_vectors);
            if (!patch_count) {
                std::cerr << "Embedding extraction failed for "
                          << entry.path.filename().string() << ": "
                          << patch_count.error().message << "\n";
                continue;
            }
            ++position_images;
            ++processed;
            std::cout << "[" << processed << "/" << total_images << "] "
                      << entry.path.filename().string() << " → "
                      << *patch_count << " patches\n";
        }

        if (position_vectors.empty()) {
            std::cerr << "Position " << pid << ": no embeddings — skipped\n";
            continue;
        }

        const auto selection_started = std::chrono::steady_clock::now();
        auto bank_result = detection::FeatureBank::BuildGreedyFromVectors(
            position_vectors, kEmbedDim, cli.coreset_max_samples);
        const auto selection_elapsed =
            std::chrono::steady_clock::now() - selection_started;
        if (!bank_result) {
            std::cerr << "Position " << pid << ": FeatureBank build failed: "
                      << bank_result.error().message << "\n";
            return 1;
        }

        auto bank_path = output_dir / ("pos_" + std::to_string(pid) + ".bin");
        const auto save_started = std::chrono::steady_clock::now();
        auto save_result = bank_result->SaveToFile(bank_path);
        const auto save_elapsed = std::chrono::steady_clock::now() - save_started;
        if (!save_result) {
            std::cerr << "Position " << pid << ": failed to save coreset: "
                      << save_result.error().message << "\n";
            return 1;
        }
        auto profile = detection::NormalityProfile::ComputeFast(*bank_result);
        auto profile_result = profile.SaveToYaml(
            bank_path.string() + ".profile.yaml");
        if (!profile_result) {
            std::cerr << "Position " << pid << ": failed to save profile: "
                      << profile_result.error().message << "\n";
            return 1;
        }

        manifest.banks.push_back(io::CoresetBankEntry{
            .position_id = pid,
            .path = bank_path,
        });

        std::cout << "Position " << pid << ":\n"
                  << "  Images:       " << position_images << "\n"
                  << "  Coreset size: " << bank_result->NumSamples() << "\n"
                  << "  Feature dim:  " << bank_result->Dim() << "\n"
                  << "  Output:       " << bank_path.string() << "\n"
                  << "  Greedy FPS:   "
                  << std::chrono::duration<double>(selection_elapsed).count() << " s\n"
                  << "  Save:         "
                  << std::chrono::duration<double>(save_elapsed).count() << " s\n\n";
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
              << "Total images:     " << processed << "\n"
              << "Timing import:    " << std::chrono::duration<double>(timing.import).count() << " s\n"
              << "Timing preprocess:" << std::chrono::duration<double>(timing.preprocess).count() << " s\n"
              << "Timing HtoD:      " << std::chrono::duration<double>(timing.upload).count() << " s\n"
              << "Timing inference: " << std::chrono::duration<double>(timing.inference).count() << " s\n"
              << "Timing DtoH:      " << std::chrono::duration<double>(timing.download).count() << " s\n"
              << "Timing total:     "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - build_started).count()
              << " s\n";
    return 0;
}
