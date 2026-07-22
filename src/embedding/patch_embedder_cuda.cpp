// patch_embedder_cuda.cpp — PatchEmbedder GPU 推理路径（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 PatchEmbedder::ExtractGpu：
//   1. 校验输入为 GpuImage
//   2. 调用 DinoV3Adapter::InferAsync() 在独立 CUDA stream 上执行推理
//   3. cudaMemcpyAsync D2D（TRT→pool，零 CPU 中转）或 D2H（回退）
//   4. cudaStreamSynchronize 等待 stream 完成后返回 Embedding
//
// 每实例持有独立 CUDA stream，避免默认流隐式串行化：
//   - 多线程推理时，各线程的 GPU kernel 和 memcpy 可真正并发执行
//   - 不再受限于 CUDA 默认流的全局顺序约束
//   - 即使单线程，后续也可在 worker loop 层实现 submit→dequeue next→sync 的
//     CPU/GPU overlap 模式

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

auto PatchEmbedder::InitializeGpuResources() noexcept -> Result<void> {
    auto stream_err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&cuda_stream_));
    if (stream_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("PatchEmbedder: failed to create CUDA stream: ")
                + cudaGetErrorString(stream_err),
        });
    }
    auto adapter_result = adapter_.Initialize();
    if (!adapter_result) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_));
        cuda_stream_ = nullptr;
        return adapter_result;
    }
    return {};
}

PatchEmbedder::~PatchEmbedder() {
    if (cuda_stream_ != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_));
        cuda_stream_ = nullptr;
    }
}

auto PatchEmbedder::ExtractGpu(const sai::image::Image& image) noexcept
    -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder::ExtractGpu: adapter has been moved away",
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_NotGpuImage,
            "PatchEmbedder::ExtractGpu requires a GpuImage",
        });
    }

    if (cuda_stream_ == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder GPU resources are not initialized",
        });
    }
    auto* stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

    const auto& gpu_img = static_cast<const sai::image::GpuImage&>(image);
    auto start = std::chrono::steady_clock::now();

    // Async GPU inference on dedicated stream — no cudaStreamSynchronize here.
    auto patch_result = adapter_.InferAsync(gpu_img, stream);
    if (!patch_result) {
        return tl::make_unexpected(patch_result.error());
    }

    // Build metadata while GPU is still running (overlap CPU work).
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = EmbeddingType::Patch;
    meta.dim = patch_result->dim;
    meta.count = patch_result->grid_h * patch_result->grid_w;
    meta.grid = {patch_result->grid_h, patch_result->grid_w};

    const std::size_t feature_bytes = meta.count * meta.dim * sizeof(float);

    // ── GpuPool path: zero-copy D2D (production) ──────────────────────
    if (gpu_pool_ != nullptr) {
        auto slab_result = gpu_pool_->Acquire(feature_bytes);
        if (!slab_result) {
            return tl::make_unexpected(slab_result.error());
        }
        auto slab = std::move(*slab_result);
        cudaError_t err = cudaMemcpyAsync(
            slab.Get(), patch_result->device_ptr, feature_bytes,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Inference_EngineExecutionFailed,
                std::string("PatchEmbedder: D2D copy failed: ")
                    + cudaGetErrorString(err),
            });
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        auto end = std::chrono::steady_clock::now();
        meta.inference_latency =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        if (sync_err != cudaSuccess) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Inference_EngineExecutionFailed,
                std::string("PatchEmbedder: stream sync failed: ")
                    + cudaGetErrorString(sync_err),
            });
        }
        return Embedding::FromGpu(std::move(slab), std::move(meta));
    }

    // Offline tools without a configured output pool consume CPU embeddings.
    std::vector<float> cpu_features(patch_result->grid_h *
                                     patch_result->grid_w *
                                     patch_result->dim);
    cudaError_t d2h_err = cudaMemcpyAsync(
        cpu_features.data(), patch_result->device_ptr,
        cpu_features.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (d2h_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("PatchEmbedder: D2H copy failed: ")
                + cudaGetErrorString(d2h_err),
        });
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    meta.inference_latency =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    if (sync_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("PatchEmbedder: stream sync failed: ")
                + cudaGetErrorString(sync_err),
        });
    }

    return Embedding::FromCpu(std::move(cpu_features), std::move(meta));
}

}  // namespace sai::embedding
