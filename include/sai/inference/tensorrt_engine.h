// tensorrt_engine.h — TensorRT 推理引擎（批次 3.1，CUDA 门控）
// 该头文件仅在目标平台（Ubuntu x64 + NVIDIA GPU）上可编译——它直接包含
// <NvInfer.h> 以使用 nvinfer1::IRuntime/ICudaEngine/IExecutionContext 的
// 完整类型。非 CUDA 构建不会包含此文件。
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>

#include <NvInfer.h>

#include <sai/inference/inference_engine.h>

namespace sai::inference {

class TensorRtEngine final : public IInferenceEngine {
public:
    explicit TensorRtEngine(std::size_t device_ordinal = 0) noexcept;
    ~TensorRtEngine() override;

    [[nodiscard]] auto Load(const std::filesystem::path& engine_path,
                             std::vector<TensorBinding> inputs,
                             std::vector<TensorBinding> outputs) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Infer() noexcept -> Result<void> override;
    [[nodiscard]] auto InferAsync(void* stream) noexcept -> Result<void> override;
    [[nodiscard]] auto Reload(const std::filesystem::path& engine_path) noexcept
        -> Result<void> override;
    [[nodiscard]] auto SetTensorAddress(const std::string& name,
                                          void* device_ptr) noexcept -> Result<void> override;
    [[nodiscard]] auto InputBindings() const noexcept -> const std::vector<TensorBinding>& override;
    [[nodiscard]] auto OutputBindings() const noexcept -> const std::vector<TensorBinding>& override;

    TensorRtEngine(const TensorRtEngine&) = delete;
    auto operator=(const TensorRtEngine&) -> TensorRtEngine& = delete;
    TensorRtEngine(TensorRtEngine&&) = delete;
    auto operator=(TensorRtEngine&&) -> TensorRtEngine& = delete;

private:
    struct EngineState {
        std::shared_ptr<nvinfer1::IRuntime> runtime;
        std::shared_ptr<nvinfer1::ICudaEngine> engine;
        std::shared_ptr<nvinfer1::IExecutionContext> context;
    };

    [[nodiscard]] auto DeserializeAndValidate(const std::filesystem::path& engine_path,
                                                const std::vector<TensorBinding>& inputs,
                                                const std::vector<TensorBinding>& outputs) noexcept
        -> Result<std::shared_ptr<EngineState>>;

    std::atomic<std::shared_ptr<EngineState>> state_{nullptr};
    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;
    std::size_t device_ordinal_;
};

}  // namespace sai::inference
