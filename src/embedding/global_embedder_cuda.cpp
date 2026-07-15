// global_embedder_cuda.cpp — GlobalEmbedder GPU 推理路径（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 GlobalEmbedder::ExtractGpu 的 GPU 推理路径：
//   1. 校验输入为 GpuImage
//   2. 调用 ClipAdapter::Infer() 获取 GlobalFeatures（GPU 指针）
//   3. cudaMemcpy 至 CPU，构造 Embedding::FromCpu()
//
// CLIP 输出单个全局特征向量（[CLS] token），维度较小（512/768），
// DtoH 拷贝开销可忽略。

#include <sai/embedding/embedder.h>

#include <chrono>
#include <source_location>
#include <vector>

#include <cuda_runtime.h>

#include <sai/image/gpu_image.h>
#include <sai/inference/clip_adapter.h>

namespace sai::embedding {

auto GlobalEmbedder::ExtractGpu(const sai::image::Image& image) noexcept
    -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "GlobalEmbedder::ExtractGpu: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Embedding_NotGpuImage,
            .message = "GlobalEmbedder::ExtractGpu requires a GpuImage",
            .source_location = std::source_location::current(),
        });
    }

    const auto& gpu_img = static_cast<const sai::image::GpuImage&>(image);
    auto start = std::chrono::steady_clock::now();

    // Run CLIP inference on GPU.
    auto global_result = adapter_.Infer(gpu_img);
    if (!global_result) {
        return tl::make_unexpected(global_result.error());
    }

    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Download CLIP [CLS] feature to CPU (single vector, small — 512/768 floats).
    std::vector<float> cpu_features(global_result->dim);
    cudaMemcpy(cpu_features.data(), global_result->device_ptr,
               global_result->dim * sizeof(float), cudaMemcpyDeviceToHost);

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
