// clip_adapter.cpp — CLIP Adapter 推理实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台上编译。当前仅实现 image encoder 路径，
// text encoder 不在 M3 范围内（见 spec §1.4 排除项）。

#include <sai/inference/clip_adapter.h>

#include <sai/image/gpu_image.h>

#include <string>

namespace sai::inference {

auto ClipAdapter::Infer(const sai::image::GpuImage& image) noexcept -> Result<GlobalFeatures> {
    // 1. 设置 "pixel_values" 输入 tensor 的 GPU 地址。
    auto set_result = engine_->SetTensorAddress(
        "pixel_values",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. 执行同步 GPU 推理。
    auto infer_result = engine_->Infer();
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 3. 读取 "image_features" 输出 binding。
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "features") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "ClipAdapter: output binding 'features' not found",
        });
    }

    if (features_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "ClipAdapter: output binding 'features' has null device_ptr",
        });
    }

    // 4. 校验输出维度与配置一致。
    std::size_t expected_bytes = cfg_.embed_dim * sizeof(float);
    if (features_binding->size_bytes < expected_bytes) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "ClipAdapter: output size mismatch (expected " +
                       std::to_string(expected_bytes) + " bytes, got " +
                       std::to_string(features_binding->size_bytes) + ")",
        });
    }

    return GlobalFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr),
        .dim = cfg_.embed_dim,
    };
}

auto ClipAdapter::InferAsync(const sai::image::GpuImage& image,
                              void* stream) noexcept -> Result<GlobalFeatures> {
    // 1. Set input tensor address (same as sync path)
    auto set_result = engine_->SetTensorAddress(
        "pixel_values",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. Execute async GPU inference — no cudaStreamSynchronize.
    auto infer_result = engine_->InferAsync(stream);
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 3. Read output binding (same as sync path)
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "features") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "ClipAdapter: output binding 'features' not found",
        });
    }

    if (features_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "ClipAdapter: output binding 'features' has null device_ptr",
        });
    }

    // 4. Validate output dimensions
    std::size_t expected_bytes = cfg_.embed_dim * sizeof(float);
    if (features_binding->size_bytes < expected_bytes) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "ClipAdapter: output size mismatch (expected " +
                       std::to_string(expected_bytes) + " bytes, got " +
                       std::to_string(features_binding->size_bytes) + ")",
        });
    }

    return GlobalFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr),
        .dim = cfg_.embed_dim,
    };
}

}  // namespace sai::inference
