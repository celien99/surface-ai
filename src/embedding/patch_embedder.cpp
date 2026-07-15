// patch_embedder.cpp — PatchEmbedder 可移植实现（批次 3.2）
// Extract/ExtractBatch 的 GPU 推理路径定义在 patch_embedder_cuda.cpp（CUDA 门控）。
// 本文件提供 GPU guard + 可移植 stub。
#include <sai/embedding/embedder.h>

namespace sai::embedding {

PatchEmbedder::PatchEmbedder(sai::inference::DinoV3Adapter adapter) noexcept
    : adapter_(std::move(adapter)), has_adapter_(true) {}

PatchEmbedder::PatchEmbedder(PatchEmbedder&& other) noexcept
    : adapter_(std::move(other.adapter_)), has_adapter_(true)
{
    other.has_adapter_ = false;
}

auto PatchEmbedder::operator=(PatchEmbedder&& other) noexcept -> PatchEmbedder& {
    if (this != &other) {
        adapter_ = std::move(other.adapter_);
        has_adapter_ = true;
        other.has_adapter_ = false;
    }
    return *this;
}

auto PatchEmbedder::Create(sai::inference::DinoV3Adapter adapter) noexcept
    -> Result<PatchEmbedder> {
    // Adapter 已由 DinoV3Adapter::Create 完成 binding 校验。
    // PatchEmbedder::Create 仅做包装，不额外校验。
    return PatchEmbedder{std::move(adapter)};
}

auto PatchEmbedder::Extract(const sai::image::Image& image) noexcept -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "PatchEmbedder::Extract: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Embedding_NotGpuImage,
            .message = "PatchEmbedder::Extract requires a GpuImage, "
                       "but received a CPU-backed image",
            .source_location = std::source_location::current(),
        });
    }

    // Delegate to GPU inference path.
    // On CUDA builds, ExtractGpu is defined in patch_embedder_cuda.cpp.
    // On non-CUDA builds, the stub below returns an error.
    return ExtractGpu(image);
}

auto PatchEmbedder::ExtractBatch(
    std::span<const sai::image::Image* const> images) noexcept
    -> Result<std::vector<Embedding>> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "PatchEmbedder::ExtractBatch: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    for (const auto* image_ptr : images) {
        if (!image_ptr || !image_ptr->IsGpuImage()) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Embedding_NotGpuImage,
                .message = "PatchEmbedder::ExtractBatch requires all images to be GpuImage",
                .source_location = std::source_location::current(),
            });
        }
    }

    // GPU 推理路径：定义在 patch_embedder_cuda.cpp（CUDA 门控）。
    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_EngineExecutionFailed,
        .message = "PatchEmbedder::ExtractBatch: GPU inference not available on this platform",
        .source_location = std::source_location::current(),
    });
}

// Non-CUDA stub for ExtractGpu — overridden by patch_embedder_cuda.cpp
// when SAI_CUDA_ENABLED is defined.
#if !defined(SAI_CUDA_ENABLED)
auto PatchEmbedder::ExtractGpu(const sai::image::Image& /*image*/) noexcept
    -> Result<Embedding> {
    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_EngineExecutionFailed,
        .message = "PatchEmbedder::ExtractGpu: GPU inference not available on this platform",
        .source_location = std::source_location::current(),
    });
}
#endif

}  // namespace sai::embedding
