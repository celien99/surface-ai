// patch_embedder.cpp — PatchEmbedder 可移植实现（批次 3.2）
// Extract/ExtractBatch 的 GPU 推理路径定义在 patch_embedder_cuda.cpp（CUDA 门控）。
// 本文件提供 GPU guard + 可移植 stub。
#include <sai/embedding/embedder.h>

namespace sai::embedding {

PatchEmbedder::PatchEmbedder(sai::inference::DinoV3Adapter adapter) noexcept
    : adapter_(std::move(adapter)), has_adapter_(true) {}

PatchEmbedder::PatchEmbedder(PatchEmbedder&& other) noexcept
    : adapter_(std::move(other.adapter_))
    , gpu_pool_(other.gpu_pool_)
    , cuda_stream_(other.cuda_stream_)
    , has_adapter_(true)
{
    other.has_adapter_ = false;
    other.cuda_stream_ = nullptr;
}

auto PatchEmbedder::operator=(PatchEmbedder&& other) noexcept -> PatchEmbedder& {
    if (this != &other) {
        adapter_ = std::move(other.adapter_);
        gpu_pool_ = other.gpu_pool_;
        cuda_stream_ = other.cuda_stream_;
        has_adapter_ = true;
        other.has_adapter_ = false;
        other.cuda_stream_ = nullptr;
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
            ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder::Extract: adapter has been moved away",
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_NotGpuImage,
            "PatchEmbedder::Extract requires a GpuImage, "
            "but received a CPU-backed image",
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
            ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder::ExtractBatch: adapter has been moved away",
        });
    }

    for (const auto* image_ptr : images) {
        if (!image_ptr || !image_ptr->IsGpuImage()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Embedding_NotGpuImage,
                "PatchEmbedder::ExtractBatch requires all images to be GpuImage",
            });
        }
    }

    // GPU 推理路径：定义在 patch_embedder_cuda.cpp（CUDA 门控）。
    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Inference_EngineExecutionFailed,
        "PatchEmbedder::ExtractBatch: GPU inference not available on this platform",
    });
}

// Non-CUDA stub for ExtractGpu — overridden by patch_embedder_cuda.cpp
// when SAI_CUDA_ENABLED is defined.
#if !defined(SAI_CUDA_ENABLED)
auto PatchEmbedder::ExtractGpu(const sai::image::Image& /*image*/) noexcept
    -> Result<Embedding> {
    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Inference_EngineExecutionFailed,
        "PatchEmbedder::ExtractGpu: GPU inference not available on this platform",
    });
}

// 可移植析构函数：无 CUDA 环境下 cuda_stream_ 恒为 nullptr（ExtractGpu 永不被调用）
PatchEmbedder::~PatchEmbedder() = default;
#endif

}  // namespace sai::embedding
