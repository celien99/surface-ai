// onnx_engine.h — ONNX Runtime CPU inference engine
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <sai/inference/inference_engine.h>

// Forward-declare ONNX Runtime types to avoid header leakage.
// The actual Ort API headers are only included in the .cpp file.
struct OrtEnv;
struct OrtSession;
struct OrtMemoryInfo;

namespace sai::inference {

// OnnxRuntimeEngine: CPU inference via Microsoft ONNX Runtime.
//
// Implements IInferenceEngine for models exported to .onnx format.
// Provides a CPU fallback when CUDA/TensorRT is unavailable on Linux,
// or when running on non-GPU hardware. Uses Ort::Session internally.
//
// Supported models: DINOv3 ViT, CLIP ViT, SAM2 (any model exported
// via tools/export_*.py with ONNX opset 17).
//
// Performance: ~2-5 fps for DINOv3 ViT-B/14 on a modern x86-64 CPU
// (vs ~30+ fps on NVIDIA A10 via TensorRT). Acceptable for offline
// batch processing or low-throughput inspection lines.
//
// Usage:
//   OnnxRuntimeEngine engine;
//   engine.Load("dino_v3.onnx", inputs, outputs);
//   engine.Infer();  // blocking CPU inference

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
    OnnxRuntimeEngine();
    ~OnnxRuntimeEngine() override;

    [[nodiscard]] auto Load(const std::filesystem::path& model_path,
                             std::vector<TensorBinding> inputs,
                             std::vector<TensorBinding> outputs) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Infer() noexcept -> Result<void> override;
    [[nodiscard]] auto InferAsync(void* /*stream*/) noexcept -> Result<void> override;
    [[nodiscard]] auto Reload(const std::filesystem::path& model_path) noexcept
        -> Result<void> override;
    [[nodiscard]] auto SetTensorAddress(const std::string& name,
                                          void* device_ptr) noexcept -> Result<void> override;
    [[nodiscard]] auto InputBindings() const noexcept
        -> const std::vector<TensorBinding>& override;
    [[nodiscard]] auto OutputBindings() const noexcept
        -> const std::vector<TensorBinding>& override;

    OnnxRuntimeEngine(const OnnxRuntimeEngine&) = delete;
    auto operator=(const OnnxRuntimeEngine&) -> OnnxRuntimeEngine& = delete;
    OnnxRuntimeEngine(OnnxRuntimeEngine&&) = delete;
    auto operator=(OnnxRuntimeEngine&&) -> OnnxRuntimeEngine& = delete;

private:
    // Opaque ONNX Runtime session — implementation detail.
    std::unique_ptr<OrtEnv> env_;
    std::unique_ptr<OrtSession> session_;
    std::unique_ptr<OrtMemoryInfo> memory_info_;

    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::filesystem::path model_path_;
};

}  // namespace sai::inference
