// onnx_engine.cpp — ONNX Runtime CPU inference implementation
//
// Uses Microsoft ONNX Runtime C++ API (Ort::Session, Ort::Env) for
// CPU-based model inference. Compiled only when onnxruntime is found
// via CMake (find_package(onnxruntime CONFIG QUIET)).
//
// Data flow:
//   1. Load .onnx model file → Ort::Session
//   2. Infer(): copy input tensors from TensorBinding → Ort::Value
//   3. Run session → Ort::Value outputs
//   4. Copy output tensors back → TensorBinding device_ptr

#include <sai/inference/onnx_engine.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <source_location>
#include <string>
#include <vector>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <sai/core/error.h>
#include <sai/inference/inference_engine.h>

namespace sai::inference {

OnnxRuntimeEngine::OnnxRuntimeEngine() = default;

OnnxRuntimeEngine::~OnnxRuntimeEngine() {
    // Unique pointers auto-destroy in reverse order:
    // session_ → env_ (ORT requires this order)
    session_.reset();
    memory_info_.reset();
    env_.reset();
}

auto OnnxRuntimeEngine::Load(const std::filesystem::path& model_path,
                              std::vector<TensorBinding> inputs,
                              std::vector<TensorBinding> outputs) noexcept
    -> Result<void> {
    try {
        // Create ONNX Runtime environment
        auto* raw_env = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                      "SurfaceAI");
        env_.reset(raw_env);

        // Create session options
        Ort::SessionOptions session_opts;
        session_opts.SetIntraOpNumThreads(4);
        session_opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model
        auto* raw_session = new Ort::Session(*env_, model_path.c_str(), session_opts);
        session_.reset(raw_session);

        // Create CPU memory info
        auto* raw_mem = new Ort::MemoryInfo(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        memory_info_.reset(raw_mem);

    } catch (const Ort::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineLoadFailed,
            std::string("ONNX: failed to load model: ") + e.what(),
            std::source_location::current(),
        });
    }

    inputs_ = std::move(inputs);
    outputs_ = std::move(outputs);
    model_path_ = model_path;

    // Extract input/output names from session
    Ort::AllocatorWithDefaultOptions allocator;
    auto n_inputs = session_->GetInputCount();
    input_names_.reserve(n_inputs);
    for (std::size_t i = 0; i < n_inputs; ++i) {
        auto* name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }

    auto n_outputs = session_->GetOutputCount();
    output_names_.reserve(n_outputs);
    for (std::size_t i = 0; i < n_outputs; ++i) {
        auto* name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }

    return {};
}

auto OnnxRuntimeEngine::Infer() noexcept -> Result<void> {
    if (session_ == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "ONNX: no model loaded",
            std::source_location::current(),
        });
    }

    if (inputs_.empty() || outputs_.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "ONNX: no bindings configured",
            std::source_location::current(),
        });
    }

    try {
        // Build input Ort::Value tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> in_name_ptrs;

        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            const auto& binding = inputs_[i];
            if (binding.device_ptr == nullptr) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Inference_InvalidBinding,
                    "ONNX: null input binding: " + binding.name,
                    std::source_location::current(),
                });
            }

            // Build shape from binding
            std::vector<std::int64_t> shape(binding.shape.begin(), binding.shape.end());

            auto tensor = Ort::Value::CreateTensor<float>(
                *memory_info_,
                static_cast<float*>(binding.device_ptr),
                binding.size_bytes / sizeof(float),
                shape.data(),
                shape.size());

            input_tensors.push_back(std::move(tensor));
            in_name_ptrs.push_back(input_names_[i].c_str());
        }

        // Pre-allocate output name pointers
        std::vector<const char*> out_name_ptrs;
        out_name_ptrs.reserve(output_names_.size());
        for (auto& name : output_names_) {
            out_name_ptrs.push_back(name.c_str());
        }

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            in_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
            out_name_ptrs.data(), out_name_ptrs.size());

        // Copy output tensors back to bindings
        for (std::size_t i = 0; i < output_tensors.size() && i < outputs_.size(); ++i) {
            auto& binding = outputs_[i];
            auto& tensor = output_tensors[i];
            auto* tensor_data = tensor.GetTensorMutableData<float>();
            auto tensor_size = tensor.GetTensorTypeAndShapeInfo().GetElementCount()
                               * sizeof(float);

            if (binding.device_ptr != nullptr && tensor_size <= binding.size_bytes) {
                std::memcpy(binding.device_ptr, tensor_data, tensor_size);
            } else if (binding.device_ptr != nullptr) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Inference_InvalidBinding,
                    "ONNX: output buffer too small for binding: " + binding.name,
                    std::source_location::current(),
                });
            }
        }

    } catch (const Ort::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("ONNX inference failed: ") + e.what(),
            std::source_location::current(),
        });
    }

    return {};
}

auto OnnxRuntimeEngine::InferAsync(void* /*stream*/) noexcept -> Result<void> {
    // ONNX Runtime CPU mode: async is the same as sync.
    // For GPU execution provider, this would use CUDA streams.
    return Infer();
}

auto OnnxRuntimeEngine::Reload(const std::filesystem::path& model_path) noexcept
    -> Result<void> {
    // Reuse current bindings, just reload model
    auto inputs = std::move(inputs_);
    auto outputs = std::move(outputs_);
    session_.reset();
    return Load(model_path, std::move(inputs), std::move(outputs));
}

auto OnnxRuntimeEngine::SetTensorAddress(const std::string& name,
                                           void* device_ptr) noexcept -> Result<void> {
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
        ErrorCode::Inference_InvalidBinding,
        "ONNX: binding '" + name + "' not found",
        std::source_location::current(),
    });
}

auto OnnxRuntimeEngine::InputBindings() const noexcept
    -> const std::vector<TensorBinding>& {
    return inputs_;
}

auto OnnxRuntimeEngine::OutputBindings() const noexcept
    -> const std::vector<TensorBinding>& {
    return outputs_;
}

}  // namespace sai::inference
