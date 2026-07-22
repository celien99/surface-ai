// global_embedder_cuda.cpp — GlobalEmbedder GPU 推理路径（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 GlobalEmbedder::ExtractGpu 的 GPU 推理路径：
//   1. 校验输入为 GpuImage
//   2. 调用 ClipAdapter::InferAsync() 在独立 CUDA stream 上执行推理
//   3. cudaMemcpyAsync 至 CPU，构造 Embedding::FromCpu()
//
// CLIP 输出单个全局特征向量（[CLS] token），维度较小（512/768），
// DtoH 拷贝开销可忽略。
// 每实例持有独立 CUDA stream，避免默认流隐式串行化。

#include <sai/embedding/embedder.h>

#include <chrono>
#include <source_location>
#include <vector>

#include <cuda_runtime.h>

#include <sai/image/gpu_image.h>
#include <sai/inference/clip_adapter.h>

namespace sai::embedding {

auto GlobalEmbedder::InitializeGpuResources() noexcept -> Result<void> {
    auto stream_err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&cuda_stream_));
    if (stream_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("GlobalEmbedder: failed to create CUDA stream: ")
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

GlobalEmbedder::~GlobalEmbedder() {
    if (cuda_stream_ != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_));
        cuda_stream_ = nullptr;
    }
}

auto GlobalEmbedder::ExtractGpu(const sai::image::Image& image) noexcept
    -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "GlobalEmbedder::ExtractGpu: adapter has been moved away",
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_NotGpuImage,
            "GlobalEmbedder::ExtractGpu requires a GpuImage",
        });
    }

    if (cuda_stream_ == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "GlobalEmbedder GPU resources are not initialized",
        });
    }
    auto* stream = reinterpret_cast<cudaStream_t>(cuda_stream_);

    const auto& gpu_img = static_cast<const sai::image::GpuImage&>(image);
    auto start = std::chrono::steady_clock::now();

    // Async CLIP inference on dedicated stream.
    auto global_result = adapter_.InferAsync(gpu_img, stream);
    if (!global_result) {
        return tl::make_unexpected(global_result.error());
    }

    // Async D2H copy on same stream for CLIP [CLS] feature (small, 512/768 floats).
    std::vector<float> cpu_features(global_result->dim);
    cudaError_t d2h_err = cudaMemcpyAsync(
        cpu_features.data(), global_result->device_ptr,
        global_result->dim * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (d2h_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("GlobalEmbedder: D2H copy failed: ")
                + cudaGetErrorString(d2h_err),
        });
    }

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    if (sync_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("GlobalEmbedder: stream sync failed: ")
                + cudaGetErrorString(sync_err),
        });
    }

    EmbeddingMeta meta;
    meta.model_name = "CLIP";
    meta.type = EmbeddingType::Global;
    meta.dim = global_result->dim;
    meta.count = 1;
    meta.grid = {0, 0};
    meta.inference_latency = latency;

    return Embedding::FromCpu(std::move(cpu_features), std::move(meta));
}

}  // namespace sai::embedding
