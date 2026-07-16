// global_embedder.cpp — GlobalEmbedder 可移植实现（批次 3.2）
// Extract/ExtractBatch 的 GPU 推理路径定义在 global_embedder_cuda.cpp（CUDA 门控）。
// 本文件提供 GPU guard + 可移植 stub。
#include <sai/embedding/embedder.h>

namespace sai::embedding {

GlobalEmbedder::GlobalEmbedder(sai::inference::ClipAdapter adapter) noexcept
    : adapter_(std::move(adapter)), has_adapter_(true) {}

GlobalEmbedder::GlobalEmbedder(GlobalEmbedder&& other) noexcept
    : adapter_(std::move(other.adapter_))
    , cuda_stream_(other.cuda_stream_)
    , has_adapter_(true)
{
    other.has_adapter_ = false;
    other.cuda_stream_ = nullptr;
}

auto GlobalEmbedder::operator=(GlobalEmbedder&& other) noexcept -> GlobalEmbedder& {
    if (this != &other) {
        adapter_ = std::move(other.adapter_);
        cuda_stream_ = other.cuda_stream_;
        has_adapter_ = true;
        other.has_adapter_ = false;
        other.cuda_stream_ = nullptr;
    }
    return *this;
}

auto GlobalEmbedder::Create(sai::inference::ClipAdapter adapter) noexcept
    -> Result<GlobalEmbedder> {
    // Adapter 已由 ClipAdapter::Create 完成 binding 校验。
    // GlobalEmbedder::Create 仅做包装，不额外校验。
    return GlobalEmbedder{std::move(adapter)};
}

auto GlobalEmbedder::Extract(const sai::image::Image& image) noexcept -> Result<Embedding> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "GlobalEmbedder::Extract: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    if (!image.IsGpuImage()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Embedding_NotGpuImage,
            .message = "GlobalEmbedder::Extract requires a GpuImage, "
                       "but received a CPU-backed image",
            .source_location = std::source_location::current(),
        });
    }

    // Delegate to GPU inference path.
    return ExtractGpu(image);
}

auto GlobalEmbedder::ExtractBatch(
    std::span<const sai::image::Image* const> images) noexcept
    -> Result<std::vector<Embedding>> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "GlobalEmbedder::ExtractBatch: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    for (const auto* image_ptr : images) {
        if (!image_ptr || !image_ptr->IsGpuImage()) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Embedding_NotGpuImage,
                .message = "GlobalEmbedder::ExtractBatch requires all images to be GpuImage",
                .source_location = std::source_location::current(),
            });
        }
    }

    // GPU 推理路径：定义在 global_embedder_cuda.cpp（CUDA 门控）。
    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_EngineExecutionFailed,
        .message = "GlobalEmbedder::ExtractBatch: GPU inference not available on this platform",
        .source_location = std::source_location::current(),
    });
}

// Non-CUDA stub for ExtractGpu — overridden by global_embedder_cuda.cpp
// when SAI_CUDA_ENABLED is defined.
#if !defined(SAI_CUDA_ENABLED)
auto GlobalEmbedder::ExtractGpu(const sai::image::Image& /*image*/) noexcept
    -> Result<Embedding> {
    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_EngineExecutionFailed,
        .message = "GlobalEmbedder::ExtractGpu: GPU inference not available on this platform",
        .source_location = std::source_location::current(),
    });
}

// 可移植析构函数：无 CUDA 环境下 cuda_stream_ 恒为 nullptr
GlobalEmbedder::~GlobalEmbedder() = default;
#endif

}  // namespace sai::embedding
