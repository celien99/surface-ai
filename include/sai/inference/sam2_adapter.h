// sam2_adapter.h — SAM2 模型 Adapter（批次 3.1）
#pragma once

#include <cstddef>
#include <filesystem>

#include <sai/core/error.h>
#include <sai/inference/inference_engine.h>

namespace sai::image {
class GpuImage;
}  // namespace sai::image

namespace sai::inference {

struct Sam2Config {
    std::filesystem::path engine_path;
    std::size_t image_size = 1024;
};

struct SegmentationMask {
    float* device_ptr = nullptr;
    std::size_t height = 0;
    std::size_t width = 0;
};

// 当前 Infer 签名为占位——仅支持 mask prompt（const GpuImage&）。
// M5 将扩展为 variant<PointPrompt, BoxPrompt, MaskPrompt> 以覆盖 SAM2 的三种 prompt 类型。
class Sam2Adapter {
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const Sam2Config& cfg) noexcept -> Result<Sam2Adapter>;

    // 声明于此，定义留在门控 Task 3 的 .cpp 文件中。
    [[nodiscard]] auto Infer(const sai::image::GpuImage& image,
                              const sai::image::GpuImage& prompt) noexcept
        -> Result<SegmentationMask>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view { return "SAM2"; }

    Sam2Adapter(Sam2Adapter&&) noexcept = default;
    auto operator=(Sam2Adapter&&) noexcept -> Sam2Adapter& = default;
    Sam2Adapter(const Sam2Adapter&) = delete;
    auto operator=(const Sam2Adapter&) -> Sam2Adapter& = delete;

private:
    Sam2Adapter(IInferenceEngine* engine, Sam2Config cfg) noexcept;
    IInferenceEngine* engine_ = nullptr;
    Sam2Config cfg_{};
};

inline auto Sam2Adapter::Create(IInferenceEngine& engine,
                                 const Sam2Config& cfg) noexcept -> Result<Sam2Adapter> {
    const auto& inputs = engine.InputBindings();
    const auto& outputs = engine.OutputBindings();

    if (inputs.empty() && outputs.empty()) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "Sam2Adapter: engine has no bindings"});
    }

    // 校验必需 input bindings
    bool has_image = false;
    bool has_prompt = false;
    for (const auto& b : inputs) {
        if (b.name == "image") has_image = true;
        if (b.name == "prompt") has_prompt = true;
    }
    if (!has_image || !has_prompt) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "Sam2Adapter: engine missing 'image' or 'prompt' input bindings"});
    }

    // 校验必需 output binding
    bool has_masks = false;
    for (const auto& b : outputs) {
        if (b.name == "masks") has_masks = true;
    }
    if (!has_masks) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "Sam2Adapter: engine missing 'masks' output binding"});
    }

    return Sam2Adapter{&engine, cfg};
}

inline Sam2Adapter::Sam2Adapter(IInferenceEngine* engine, Sam2Config cfg) noexcept
    : engine_(engine), cfg_(std::move(cfg)) {}

}  // namespace sai::inference
