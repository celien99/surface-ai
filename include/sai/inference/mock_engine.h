// mock_engine.h — 可移植 Mock 推理引擎（批次 3.1）
#pragma once

#include <functional>
#include <sai/inference/inference_engine.h>

namespace sai::inference {

class MockEngine final : public IInferenceEngine {
public:
    // 测试数据注入：每次 Infer()/InferAsync() 调用后，对每个 output binding 调用此回调。
    // 回调签名为 void(std::string_view name, void* device_ptr, std::size_t size_bytes)，
    // 测试代码将预期数据 memcpy 到 device_ptr 中，模拟 TensorRT 的推理输出。
    using OutputFillCallback =
        std::function<void(std::string_view name, void* device_ptr, std::size_t size_bytes)>;

    auto SetOutputFillCallback(OutputFillCallback callback) noexcept -> void;

    [[nodiscard]] auto Load(const std::filesystem::path&,
                             std::vector<TensorBinding> inputs,
                             std::vector<TensorBinding> outputs) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Infer() noexcept -> Result<void> override;
    [[nodiscard]] auto InferAsync(void*) noexcept -> Result<void> override;
    [[nodiscard]] auto Reload(const std::filesystem::path&) noexcept -> Result<void> override;
    [[nodiscard]] auto SetTensorAddress(const std::string&, void*) noexcept -> Result<void> override;
    [[nodiscard]] auto InputBindings() const noexcept -> const std::vector<TensorBinding>& override;
    [[nodiscard]] auto OutputBindings() const noexcept -> const std::vector<TensorBinding>& override;

private:
    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;
    OutputFillCallback output_fill_;
};

}  // namespace sai::inference
