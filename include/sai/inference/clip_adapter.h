// clip_adapter.h — CLIP 模型 Adapter（批次 3.1，仅 image encoder）
#pragma once

#include <cstddef>
#include <filesystem>

#include <sai/core/error.h>
#include <sai/inference/inference_engine.h>

namespace sai::image {
class GpuImage;
}  // namespace sai::image

namespace sai::inference {

struct ClipConfig {
    std::filesystem::path engine_path;
    std::size_t image_size = 224;
    std::size_t embed_dim = 512;
};

struct GlobalFeatures {
    float* device_ptr = nullptr;
    std::size_t dim = 0;
};

class ClipAdapter {
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const ClipConfig& cfg) noexcept -> Result<ClipAdapter>;

    // 声明于此，定义留在门控 Task 3 的 .cpp 文件中。
    [[nodiscard]] auto Infer(const sai::image::GpuImage& image) noexcept -> Result<GlobalFeatures>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view { return "CLIP"; }

    ClipAdapter(ClipAdapter&&) noexcept = default;
    auto operator=(ClipAdapter&&) noexcept -> ClipAdapter& = default;
    ClipAdapter(const ClipAdapter&) = delete;
    auto operator=(const ClipAdapter&) -> ClipAdapter& = delete;

private:
    ClipAdapter(IInferenceEngine* engine, ClipConfig cfg) noexcept;
    IInferenceEngine* engine_ = nullptr;
    ClipConfig cfg_{};
};

inline auto ClipAdapter::Create(IInferenceEngine& engine,
                                 const ClipConfig& cfg) noexcept -> Result<ClipAdapter> {
    const auto& outputs = engine.OutputBindings();
    if (outputs.empty()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "ClipAdapter: engine has no output bindings",
            .source_location = std::source_location::current(),
        });
    }

    // 查找 "features" 输出 binding
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "features") {
            features_binding = &b;
            break;
        }
    }
    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "ClipAdapter: engine missing 'features' output binding",
            .source_location = std::source_location::current(),
        });
    }

    // 校验 embed_dim（取 shape 最后一维）
    if (features_binding->shape.empty()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "ClipAdapter: 'features' binding has empty shape",
            .source_location = std::source_location::current(),
        });
    }
    auto binding_embed_dim = static_cast<std::size_t>(features_binding->shape.back());
    if (binding_embed_dim != cfg.embed_dim) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "ClipAdapter: embed_dim mismatch (config=" +
                       std::to_string(cfg.embed_dim) +
                       ", engine=" + std::to_string(binding_embed_dim) + ")",
            .source_location = std::source_location::current(),
        });
    }

    return ClipAdapter{&engine, cfg};
}

inline ClipAdapter::ClipAdapter(IInferenceEngine* engine, ClipConfig cfg) noexcept
    : engine_(engine), cfg_(std::move(cfg)) {}

}  // namespace sai::inference
