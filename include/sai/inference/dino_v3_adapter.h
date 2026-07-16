// dino_v3_adapter.h — DINOv3 模型 Adapter（批次 3.1）
#pragma once

#include <cstddef>
#include <filesystem>

#include <sai/core/error.h>
#include <sai/inference/inference_engine.h>

namespace sai::image {
class GpuImage;
}  // namespace sai::image

namespace sai::inference {

struct DinoV3Config {
    std::filesystem::path engine_path;
    std::size_t image_size = 518;
    std::size_t patch_size = 14;
    std::size_t embed_dim = 1024;
};

struct PatchFeatures {
    float* device_ptr = nullptr;
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;
    std::size_t dim = 0;
};

class DinoV3Adapter {
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const DinoV3Config& cfg) noexcept -> Result<DinoV3Adapter>;

    // 从 M2 的 GpuImage 提取 patch 特征——该图像必须在 GPU 显存中。
    // 声明于此，定义留在门控 Task 3 的 .cpp 文件中。
    [[nodiscard]] auto Infer(const sai::image::GpuImage& image) noexcept -> Result<PatchFeatures>;

    // Async inference: enqueues work on the given CUDA stream and returns
    // immediately WITHOUT synchronizing. The caller must call
    // cudaStreamSynchronize(stream) before reading the output PatchFeatures
    // device_ptr. Opt-in path for GPU-CPU overlap scenarios.
    [[nodiscard]] auto InferAsync(const sai::image::GpuImage& image,
                                   void* stream) noexcept -> Result<PatchFeatures>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view { return "DINOv3"; }

    DinoV3Adapter(DinoV3Adapter&&) noexcept = default;
    auto operator=(DinoV3Adapter&&) noexcept -> DinoV3Adapter& = default;
    DinoV3Adapter(const DinoV3Adapter&) = delete;
    auto operator=(const DinoV3Adapter&) -> DinoV3Adapter& = delete;

private:
    DinoV3Adapter(IInferenceEngine* engine, DinoV3Config cfg) noexcept;
    IInferenceEngine* engine_ = nullptr;
    DinoV3Config cfg_{};
};

// Create 工厂实现：校验 engine 的 binding 名称/形状与 config 一致。
inline auto DinoV3Adapter::Create(IInferenceEngine& engine,
                                   const DinoV3Config& cfg) noexcept -> Result<DinoV3Adapter> {
    const auto& outputs = engine.OutputBindings();
    if (outputs.empty()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "DinoV3Adapter: engine has no output bindings",
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
            .message = "DinoV3Adapter: engine missing 'features' output binding",
            .source_location = std::source_location::current(),
        });
    }

    // 校验 embed_dim（取 shape 最后一维）
    if (features_binding->shape.empty()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "DinoV3Adapter: 'features' binding has empty shape",
            .source_location = std::source_location::current(),
        });
    }
    auto binding_embed_dim = static_cast<std::size_t>(features_binding->shape.back());
    if (binding_embed_dim != cfg.embed_dim) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ModelConfigMismatch,
            .message = "DinoV3Adapter: embed_dim mismatch (config=" +
                       std::to_string(cfg.embed_dim) +
                       ", engine=" + std::to_string(binding_embed_dim) + ")",
            .source_location = std::source_location::current(),
        });
    }

    return DinoV3Adapter{&engine, cfg};
}

inline DinoV3Adapter::DinoV3Adapter(IInferenceEngine* engine, DinoV3Config cfg) noexcept
    : engine_(engine), cfg_(std::move(cfg)) {}

}  // namespace sai::inference
