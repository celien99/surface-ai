// patch_embedder_cuda.cpp — PatchEmbedder GPU 推理路径（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 PatchEmbedder::ExtractGpu：
//   1. 校验输入为 GpuImage
//   2. 调用 DinoV3Adapter::Infer() 获取 PatchFeatures（GPU 指针）
//   3. 若已注入 GpuPool：从池中分配 slab → cudaMemcpy D2D（TRT→pool）
//      → Embedding::FromGpu (零 CPU 中转)
//   4. 否则回退：cudaMemcpy D2H → Embedding::FromCpu（兼容路径）
//
// 当 GpuPool 可用时，特征数据全程驻留在 GPU 显存上：
//   - 下游 FeatureBank::Search 可通过 feature_bank_cuda.cpp 的 GPU 后端
//     直接在 GPU 上搜索，无需二次搬移。

#include <sai/embedding/embedder.h>

#include <chrono>
#include <memory>
#include <source_location>

#include <cuda_runtime.h>

#include <sai/image/gpu_image.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/memory/memory_pool.h>
#include <sai/memory/pooled_ptr.h>

namespace sai::embedding {

auto PatchEmbedder::ExtractGpu(const sai::image::Image& image) noexcept
    -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "PatchEmbedder::ExtractGpu: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Embedding_NotGpuImage,
            .message = "PatchEmbedder::ExtractGpu requires a GpuImage",
            .source_location = std::source_location::current(),
        });
    }

    const auto& gpu_img = static_cast<const sai::image::GpuImage&>(image);
    auto start = std::chrono::steady_clock::now();

    // Run DINOv3 inference on GPU.
    auto patch_result = adapter_.Infer(gpu_img);
    if (!patch_result) {
        return tl::make_unexpected(patch_result.error());
    }

    auto end = std::chrono::steady_clock::now();
    auto latency =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Build metadata.
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = EmbeddingType::Patch;
    meta.dim = patch_result->dim;
    meta.count = patch_result->grid_h * patch_result->grid_w;
    meta.grid = {patch_result->grid_h, patch_result->grid_w};
    meta.inference_latency = latency;

    const std::size_t feature_bytes = meta.count * meta.dim * sizeof(float);

    // ── GpuPool path: zero-copy D2D (production) ──────────────────────
    if (gpu_pool_ != nullptr) {
        auto slab_result = gpu_pool_->Acquire(feature_bytes);
        if (!slab_result) {
            // Pool exhausted — fall through to CPU path rather than failing
        } else {
            auto slab = std::move(*slab_result);
            cudaError_t err = cudaMemcpy(
                slab.Get(), patch_result->device_ptr, feature_bytes,
                cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                // D2D copy failed — slab auto-released, fall through
            } else {
                // Success: feature data lives in pool-backed GPU memory.
                // Downstream FeatureBank::Search (GPU path) can consume
                // this directly without a host round-trip.
                return Embedding::FromGpu(std::move(slab), std::move(meta));
            }
        }
    }

    // ── CPU fallback: D2H copy (compatible, not zero-copy) ────────────
    std::vector<float> cpu_features(patch_result->grid_h *
                                     patch_result->grid_w *
                                     patch_result->dim);
    cudaMemcpy(cpu_features.data(), patch_result->device_ptr,
               cpu_features.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return Embedding::FromCpu(std::move(cpu_features), std::move(meta));
}

}  // namespace sai::embedding
