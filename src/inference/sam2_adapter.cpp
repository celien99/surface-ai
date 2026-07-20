// sam2_adapter.cpp — SAM2 Adapter 推理实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台上编译。Infer() 接收图像和 prompt（mask 形式），
// 输出分割掩码。

#include <sai/inference/sam2_adapter.h>

#include <sai/image/gpu_image.h>

namespace sai::inference {

auto Sam2Adapter::Infer(const sai::image::GpuImage& image,
                          const sai::image::GpuImage& prompt) noexcept
    -> Result<SegmentationMask> {
    // 1. 设置 "image" 输入 tensor 的 GPU 地址。
    auto set_result = engine_->SetTensorAddress(
        "image",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. 设置 "prompt" 输入 tensor 的 GPU 地址。
    set_result = engine_->SetTensorAddress(
        "prompt",
        const_cast<std::uint8_t*>(prompt.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 3. 执行同步 GPU 推理。
    auto infer_result = engine_->Infer();
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 4. 读取 "masks" 输出 binding。
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* masks_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "masks") {
            masks_binding = &b;
            break;
        }
    }

    if (masks_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "Sam2Adapter: output binding 'masks' not found",
        });
    }

    if (masks_binding->device_ptr == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "Sam2Adapter: output binding 'masks' has null device_ptr",
        });
    }

    // 5. 从 binding shape 解析 mask 的空间尺寸（SAM2 输出 shape 通常为
    //    [batch=1, height, width] 或 [batch=1, 1, height, width]）。
    std::size_t mask_height = cfg_.image_size;
    std::size_t mask_width = cfg_.image_size;

    if (masks_binding->shape.size() >= 2) {
        auto last = masks_binding->shape.size();
        mask_height = static_cast<std::size_t>(masks_binding->shape[last - 2]);
        mask_width = static_cast<std::size_t>(masks_binding->shape[last - 1]);
    }

    return SegmentationMask{
        .device_ptr = static_cast<float*>(masks_binding->device_ptr),
        .height = mask_height,
        .width = mask_width,
    };
}

}  // namespace sai::inference
