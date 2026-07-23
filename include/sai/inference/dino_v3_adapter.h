// dino_v3_adapter.h — DINO ViT Adapter（批次 3.1）
// Generic ViT patch feature extractor — compatible with DINOv2 and DINOv3.
#pragma once

#include <cstddef>
#include <filesystem>
#include <string_view>
#include <utility>

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
    [[nodiscard]] auto Initialize() noexcept -> Result<void>;

    // 从 M2 的 GpuImage 提取 patch 特征——该图像必须在 GPU 显存中。
    // 声明于此，定义留在门控 Task 3 的 .cpp 文件中。
    [[nodiscard]] auto Infer(const sai::image::GpuImage& image) noexcept -> Result<PatchFeatures>;

    // Async inference: enqueues work on the given CUDA stream and returns
    // immediately WITHOUT synchronizing. The caller must call
    // cudaStreamSynchronize(stream) before reading the output PatchFeatures
    // device_ptr. Opt-in path for GPU-CPU overlap scenarios.
    [[nodiscard]] auto InferAsync(const sai::image::GpuImage& image,
                                   void* stream) noexcept -> Result<PatchFeatures>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view { return "DINOv2"; }

    ~DinoV3Adapter();
    DinoV3Adapter(DinoV3Adapter&&) noexcept;
    auto operator=(DinoV3Adapter&&) noexcept -> DinoV3Adapter&;
    DinoV3Adapter(const DinoV3Adapter&) = delete;
    auto operator=(const DinoV3Adapter&) -> DinoV3Adapter& = delete;

private:
    DinoV3Adapter(IInferenceEngine* engine, DinoV3Config cfg,
                  TensorDataType input_dtype,
                  TensorDataType output_dtype) noexcept;
    [[nodiscard]] auto InferImpl(const sai::image::GpuImage& image,
                                 void* stream, bool synchronize) noexcept
        -> Result<PatchFeatures>;
    auto ReleaseBuffers() noexcept -> void;
    IInferenceEngine* engine_ = nullptr;
    DinoV3Config cfg_{};
    TensorDataType input_dtype_ = TensorDataType::Unknown;
    TensorDataType output_dtype_ = TensorDataType::Unknown;
    void* input_buffer_ = nullptr;
    void* raw_output_buffer_ = nullptr;
    float* patch_output_buffer_ = nullptr;
};

// Create 工厂实现：校验 engine 的 binding 名称/形状与 config 一致。
inline auto DinoV3Adapter::Create(IInferenceEngine& engine,
                                   const DinoV3Config& cfg) noexcept -> Result<DinoV3Adapter> {
    const auto& inputs = engine.InputBindings();
    const auto& outputs = engine.OutputBindings();
    const TensorBinding* input_binding = nullptr;
    for (const auto& b : inputs) {
        if (b.name == "pixel_values") {
            input_binding = &b;
            break;
        }
    }
    if (input_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: engine missing 'pixel_values' input binding"});
    }
    if (input_binding->shape != std::vector<std::int64_t>{
            1, 3, static_cast<std::int64_t>(cfg.image_size),
            static_cast<std::int64_t>(cfg.image_size)}) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: 'pixel_values' must have shape [1,3,H,W]"});
    }
    if (input_binding->dtype != TensorDataType::Float16 &&
        input_binding->dtype != TensorDataType::Float32) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: 'pixel_values' must be float16 or float32"});
    }
    if (outputs.empty()) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "DinoV3Adapter: engine has no output bindings"});
    }

    // 查找 "last_hidden_state" 输出 binding
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            features_binding = &b;
            break;
        }
    }
    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "DinoV3Adapter: engine missing 'last_hidden_state' output binding"});
    }
    if (features_binding->dtype != TensorDataType::Float16 &&
        features_binding->dtype != TensorDataType::Float32) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: 'last_hidden_state' must be float16 or float32"});
    }
    const auto grid = cfg.image_size / cfg.patch_size;
    if (features_binding->shape != std::vector<std::int64_t>{
            1, static_cast<std::int64_t>(grid * grid + 1),
            static_cast<std::int64_t>(cfg.embed_dim)}) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: unexpected 'last_hidden_state' shape"});
    }

    // 校验 embed_dim（取 shape 最后一维）
    if (features_binding->shape.empty()) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch, "DinoV3Adapter: 'last_hidden_state' binding has empty shape"});
    }
    auto binding_embed_dim = static_cast<std::size_t>(features_binding->shape.back());
    if (binding_embed_dim != cfg.embed_dim) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Inference_ModelConfigMismatch,
                                   "DinoV3Adapter: embed_dim mismatch (config=" +
                                   std::to_string(cfg.embed_dim) +
                                   ", engine=" + std::to_string(binding_embed_dim) + ")"});
    }

    return DinoV3Adapter{
        &engine, cfg, input_binding->dtype, features_binding->dtype};
}

inline DinoV3Adapter::DinoV3Adapter(
    IInferenceEngine* engine, DinoV3Config cfg,
    TensorDataType input_dtype, TensorDataType output_dtype) noexcept
    : engine_(engine), cfg_(std::move(cfg)),
      input_dtype_(input_dtype), output_dtype_(output_dtype) {}

}  // namespace sai::inference
