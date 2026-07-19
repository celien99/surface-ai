// mock_engine.cpp — MockEngine 实现（批次 3.1 可移植子集）
#include <sai/inference/mock_engine.h>

namespace sai::inference {

auto MockEngine::SetOutputFillCallback(OutputFillCallback callback) noexcept -> void {
    output_fill_ = std::move(callback);
}

auto MockEngine::Load(const std::filesystem::path&,
                       std::vector<TensorBinding> inputs,
                       std::vector<TensorBinding> outputs) noexcept -> Result<void> {
    inputs_ = std::move(inputs);
    outputs_ = std::move(outputs);
    return {};
}

auto MockEngine::Infer() noexcept -> Result<void> {
    if (inputs_.empty() && outputs_.empty()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "MockEngine: no bindings loaded",
            .source_location = std::source_location::current(),
        });
    }

    for (const auto& output : outputs_) {
        if (output.device_ptr == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Inference_InvalidBinding,
                .message = "MockEngine: output binding '" + output.name + "' has null device_ptr",
                .source_location = std::source_location::current(),
            });
        }
    }

    if (output_fill_) {
        for (const auto& output : outputs_) {
            output_fill_(output.name, output.device_ptr, output.size_bytes);
        }
    }

    return {};
}

auto MockEngine::InferAsync(void* /*stream*/) noexcept -> Result<void> {
    // Mock: same as Infer, no real async behavior
    return Infer();
}

auto MockEngine::Reload(const std::filesystem::path&) noexcept -> Result<void> {
    // Mock: reload is a no-op that preserves current bindings
    return {};
}

auto MockEngine::SetTensorAddress(const std::string& name, void* device_ptr) noexcept -> Result<void> {
    for (auto& binding : outputs_) {
        if (binding.name == name) {
            binding.device_ptr = device_ptr;
            return {};
        }
    }
    for (auto& binding : inputs_) {
        if (binding.name == name) {
            binding.device_ptr = device_ptr;
            return {};
        }
    }
    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_InvalidBinding,
        .message = "MockEngine: binding '" + name + "' not found",
        .source_location = std::source_location::current(),
    });
}

auto MockEngine::InputBindings() const noexcept -> const std::vector<TensorBinding>& {
    return inputs_;
}

auto MockEngine::OutputBindings() const noexcept -> const std::vector<TensorBinding>& {
    return outputs_;
}

}  // namespace sai::inference
