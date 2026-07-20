// dino_v3_adapter.cpp — DINO ViT Adapter 推理实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台上编译。Infer() 方法调用 TensorRtEngine 执行实际的 GPU 推理。
// DINOv2 输出包含 CLS token，需跳过首 token 再交 PatchEmbedder。

#include <sai/inference/dino_v3_adapter.h>

#include <sai/image/gpu_image.h>

#include <string>

namespace sai::inference {

auto DinoV3Adapter::Infer(const sai::image::GpuImage& image) noexcept -> Result<PatchFeatures> {
    // 1. 设置输入 tensor 的 GPU 地址——image.Data() 返回 const uint8_t*，
    //    但 SetTensorAddress 需要 void*（设备指针，const 对设备端无意义）。
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

    // 3. 读取输出 binding，获取 GPU 端的 patch features。
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' not found",
        });
    }

    if (features_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' has null device_ptr",
        });
    }

    // 4. 计算 patch grid 并校验维度。
    // DINOv2 的 last_hidden_state 包含 CLS token 作为首 token，
    // 因此期望大小比纯 patch 多一个 token。
    auto grid_h = cfg_.image_size / cfg_.patch_size;
    auto grid_w = cfg_.image_size / cfg_.patch_size;

    std::size_t expected_elements = (grid_h * grid_w + 1) * cfg_.embed_dim;  // +1 for CLS
    std::size_t actual_elements = features_binding->size_bytes / sizeof(float);
    if (actual_elements < expected_elements) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: output size mismatch (expected " +
                       std::to_string(expected_elements) + " floats, got " +
                       std::to_string(actual_elements) + ")",
        });
    }

    // 跳过 CLS token（首个 token），返回纯 patch features
    return PatchFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr) + cfg_.embed_dim,
        .grid_h = grid_h,
        .grid_w = grid_w,
        .dim = cfg_.embed_dim,
    };
}

auto DinoV3Adapter::InferAsync(const sai::image::GpuImage& image,
                                void* stream) noexcept -> Result<PatchFeatures> {
    // 1. Set input tensor address (same as sync path)
    auto set_result = engine_->SetTensorAddress(
        "pixel_values",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. Execute async GPU inference — no cudaStreamSynchronize.
    // The caller owns stream lifecycle and synchronization.
    auto infer_result = engine_->InferAsync(stream);
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 3. Read output binding (same as sync path)
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' not found",
        });
    }

    if (features_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' has null device_ptr",
        });
    }

    // 4. Compute patch grid and validate dimensions.
    // DINOv2's last_hidden_state includes a CLS token as the first token,
    // so expected size is one token larger than pure patches.
    auto grid_h = cfg_.image_size / cfg_.patch_size;
    auto grid_w = cfg_.image_size / cfg_.patch_size;

    std::size_t expected_elements = (grid_h * grid_w + 1) * cfg_.embed_dim;  // +1 for CLS
    std::size_t actual_elements = features_binding->size_bytes / sizeof(float);
    if (actual_elements < expected_elements) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: output size mismatch (expected " +
                       std::to_string(expected_elements) + " floats, got " +
                       std::to_string(actual_elements) + ")",
        });
    }

    // Skip CLS token (first token), return pure patch features
    return PatchFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr) + cfg_.embed_dim,
        .grid_h = grid_h,
        .grid_w = grid_w,
        .dim = cfg_.embed_dim,
    };
}

}  // namespace sai::inference
