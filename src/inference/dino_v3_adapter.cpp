// dino_v3_adapter.cpp — DINOv3 Adapter 推理实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台上编译。Infer() 方法调用 TensorRtEngine 执行实际的 GPU 推理。

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
            .code = ErrorCode::Inference_InvalidBinding,
            .message = "DinoV3Adapter: output binding 'last_hidden_state' not found",
            .source_location = std::source_location::current(),
        });
    }

    if (features_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_InvalidBinding,
            .message = "DinoV3Adapter: output binding 'last_hidden_state' has null device_ptr",
            .source_location = std::source_location::current(),
        });
    }

    // 4. 计算 patch grid 并校验维度。
    auto grid_h = cfg_.image_size / cfg_.patch_size;
    auto grid_w = cfg_.image_size / cfg_.patch_size;

    std::size_t expected_elements = grid_h * grid_w * cfg_.embed_dim;
    std::size_t actual_elements = features_binding->size_bytes / sizeof(float);
    if (actual_elements < expected_elements) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "DinoV3Adapter: output size mismatch (expected " +
                       std::to_string(expected_elements) + " floats, got " +
                       std::to_string(actual_elements) + ")",
            .source_location = std::source_location::current(),
        });
    }

    return PatchFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr),
        .grid_h = grid_h,
        .grid_w = grid_w,
        .dim = cfg_.embed_dim,
    };
}

}  // namespace sai::inference
