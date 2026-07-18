// tensorrt_engine.cpp — TensorRtEngine 实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台（Ubuntu x64 + NVIDIA GPU + TensorRT SDK）上编译。
// 可移植构建中此文件不在源文件列表中，不会触发编译错误。

#include <sai/inference/tensorrt_engine.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace sai::inference {

namespace {

// TensorRT 日志适配器——将 TRT 内部日志转发到 sai::Logger 或 stderr。
// 静态单例，与 NVIDIA 官方 samples 的 Logger 模式一致。
class TrtLogger : public nvinfer1::ILogger {
public:
    static auto Instance() noexcept -> TrtLogger& {
        static TrtLogger logger;
        return logger;
    }

    void log(Severity severity, const char* msg) noexcept override {
        // 过滤 info/verbose 级别的 TRT 内部日志，仅转发 warning 及以上。
        if (severity <= Severity::kWARNING) {
            // 在无 sai::Logger 可用的情况下回退到 stderr——TRT Logger 在
            // sai::Logger 初始化之前就可能被调用（static init 期间）。
            std::fprintf(stderr, "[TensorRT] %s: %s\n",
                         SeverityToString(severity), msg);
        }
    }

private:
    TrtLogger() noexcept = default;

    static auto SeverityToString(Severity severity) noexcept -> const char* {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR:          return "ERROR";
            case Severity::kWARNING:        return "WARNING";
            case Severity::kINFO:           return "INFO";
            case Severity::kVERBOSE:        return "VERBOSE";
            default:                        return "UNKNOWN";
        }
    }
};

// 读取 engine 文件的全部字节。
[[nodiscard]] auto ReadEngineFile(const std::filesystem::path& path) noexcept
    -> Result<std::vector<char>> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: failed to open engine file: " + path.string(),
            .source_location = std::source_location::current(),
        });
    }

    auto size = file.tellg();
    if (size <= 0) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: engine file is empty: " + path.string(),
            .source_location = std::source_location::current(),
        });
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<std::size_t>(size));
    if (!file.read(buffer.data(), size)) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: failed to read engine file: " + path.string(),
            .source_location = std::source_location::current(),
        });
    }

    return buffer;
}

// 校验 engine 的 I/O binding 名称与调用方提供的绑定列表一致。
// 返回 nullptr 表示通过，否则返回错误。
[[nodiscard]] auto ValidateBindings(nvinfer1::ICudaEngine& engine,
                                     const std::vector<TensorBinding>& inputs,
                                     const std::vector<TensorBinding>& outputs) noexcept
    -> std::optional<ErrorInfo> {
    auto num_bindings = engine.getNbIOTensors();
    auto expected_count = inputs.size() + outputs.size();
    if (static_cast<std::size_t>(num_bindings) != expected_count) {
        return ErrorInfo{
            .code = ErrorCode::Inference_InvalidBinding,
            .message = "TensorRtEngine: binding count mismatch (engine=" +
                       std::to_string(num_bindings) +
                       ", provided=" + std::to_string(expected_count) + ")",
            .source_location = std::source_location::current(),
        };
    }

    // 收集 engine 中所有 I/O tensor 名称。
    std::vector<std::string> engine_names;
    engine_names.reserve(static_cast<std::size_t>(num_bindings));
    for (std::int32_t i = 0; i < num_bindings; ++i) {
        engine_names.emplace_back(engine.getIOTensorName(i));
    }

    // 校验每个 input binding 名称存在于 engine 中。
    for (const auto& binding : inputs) {
        auto it = std::find(engine_names.begin(), engine_names.end(), binding.name);
        if (it == engine_names.end()) {
            return ErrorInfo{
                .code = ErrorCode::Inference_InvalidBinding,
                .message = "TensorRtEngine: input binding '" + binding.name +
                           "' not found in engine",
                .source_location = std::source_location::current(),
            };
        }
    }

    // 校验每个 output binding 名称存在于 engine 中。
    for (const auto& binding : outputs) {
        auto it = std::find(engine_names.begin(), engine_names.end(), binding.name);
        if (it == engine_names.end()) {
            return ErrorInfo{
                .code = ErrorCode::Inference_InvalidBinding,
                .message = "TensorRtEngine: output binding '" + binding.name +
                           "' not found in engine",
                .source_location = std::source_location::current(),
            };
        }
    }

    return std::nullopt;
}

// 根据 device_ordinal 设置当前 CUDA 设备（如果当前设备不同）。
// 返回错误若 ordinal 无效。
[[nodiscard]] auto SetCudaDevice(std::size_t device_ordinal) noexcept -> std::optional<ErrorInfo> {
    int current_device = -1;
    auto err = cudaGetDevice(&current_device);
    if (err != cudaSuccess) {
        return ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: cudaGetDevice failed: " +
                       std::string(cudaGetErrorString(err)),
            .source_location = std::source_location::current(),
        };
    }

    if (static_cast<std::size_t>(current_device) != device_ordinal) {
        int device_count = 0;
        err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            return ErrorInfo{
                .code = ErrorCode::Inference_EngineLoadFailed,
                .message = "TensorRtEngine: cudaGetDeviceCount failed: " +
                           std::string(cudaGetErrorString(err)),
                .source_location = std::source_location::current(),
            };
        }

        if (static_cast<int>(device_ordinal) >= device_count) {
            return ErrorInfo{
                .code = ErrorCode::Inference_EngineLoadFailed,
                .message = "TensorRtEngine: device ordinal " +
                           std::to_string(device_ordinal) +
                           " out of range (device count=" +
                           std::to_string(device_count) + ")",
                .source_location = std::source_location::current(),
            };
        }

        err = cudaSetDevice(static_cast<int>(device_ordinal));
        if (err != cudaSuccess) {
            return ErrorInfo{
                .code = ErrorCode::Inference_EngineLoadFailed,
                .message = "TensorRtEngine: cudaSetDevice failed: " +
                           std::string(cudaGetErrorString(err)),
                .source_location = std::source_location::current(),
            };
        }
    }

    return std::nullopt;
}

}  // namespace

TensorRtEngine::TensorRtEngine(std::size_t device_ordinal) noexcept
    : device_ordinal_(device_ordinal) {}

TensorRtEngine::~TensorRtEngine() {
    // atomic<shared_ptr<EngineState>> 在析构时自动释放最后一个引用——
    // EngineState 的 shared_ptr 成员依次析构，各自的自定义 deleter 调用
    // delete 释放 nvinfer1 对象（TRT 10+ 移除了 destroy()，改用 delete）。
    state_.store(nullptr);
}

auto TensorRtEngine::DeserializeAndValidate(
    const std::filesystem::path& engine_path,
    const std::vector<TensorBinding>& inputs,
    const std::vector<TensorBinding>& outputs) noexcept
    -> Result<std::shared_ptr<EngineState>> {
    // 1. 设置 CUDA 设备。
    if (auto err = SetCudaDevice(device_ordinal_)) {
        return tl::make_unexpected(std::move(*err));
    }

    // 2. 读取 engine 文件。
    auto file_data = ReadEngineFile(engine_path);
    if (!file_data.has_value()) {
        return tl::make_unexpected(file_data.error());
    }

    // 3. 创建 IRuntime。
    auto* raw_runtime = nvinfer1::createInferRuntime(TrtLogger::Instance());
    if (raw_runtime == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: createInferRuntime returned nullptr",
            .source_location = std::source_location::current(),
        });
    }
    auto runtime = std::shared_ptr<nvinfer1::IRuntime>(
        raw_runtime,
        [](nvinfer1::IRuntime* p) { delete p; });

    // 4. 反序列化 engine。
    auto* raw_engine = runtime->deserializeCudaEngine(
        file_data->data(), file_data->size());
    if (raw_engine == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: deserializeCudaEngine returned nullptr "
                       "(file may be corrupted or from an incompatible TensorRT version)",
            .source_location = std::source_location::current(),
        });
    }
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        raw_engine,
        [](nvinfer1::ICudaEngine* p) { delete p; });

    // 5. 创建 ExecutionContext。
    auto* raw_context = engine->createExecutionContext();
    if (raw_context == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineLoadFailed,
            .message = "TensorRtEngine: createExecutionContext returned nullptr",
            .source_location = std::source_location::current(),
        });
    }
    auto context = std::shared_ptr<nvinfer1::IExecutionContext>(
        raw_context,
        [](nvinfer1::IExecutionContext* p) { delete p; });

    // 6. 校验 binding 名称。
    if (auto err = ValidateBindings(*engine, inputs, outputs)) {
        return tl::make_unexpected(std::move(*err));
    }

    // 7. 初始化所有 I/O tensor 的地址（如果调用方已在 binding 中提供了非空
    //    device_ptr）。TRT 10 要求在 enqueueV3 之前 setTensorAddress。
    for (const auto& binding : inputs) {
        if (binding.device_ptr != nullptr) {
            if (!context->setTensorAddress(binding.name.c_str(), binding.device_ptr)) {
                return tl::make_unexpected(ErrorInfo{
                    .code = ErrorCode::Inference_InvalidBinding,
                    .message = "TensorRtEngine: setTensorAddress failed for input '" +
                               binding.name + "'",
                    .source_location = std::source_location::current(),
                });
            }
        }
    }
    for (const auto& binding : outputs) {
        if (binding.device_ptr != nullptr) {
            if (!context->setTensorAddress(binding.name.c_str(), binding.device_ptr)) {
                return tl::make_unexpected(ErrorInfo{
                    .code = ErrorCode::Inference_InvalidBinding,
                    .message = "TensorRtEngine: setTensorAddress failed for output '" +
                               binding.name + "'",
                    .source_location = std::source_location::current(),
                });
            }
        }
    }

    auto state = std::make_shared<EngineState>();
    state->runtime = std::move(runtime);
    state->engine = std::move(engine);
    state->context = std::move(context);
    return state;
}

auto TensorRtEngine::Load(const std::filesystem::path& engine_path,
                           std::vector<TensorBinding> inputs,
                           std::vector<TensorBinding> outputs) noexcept -> Result<void> {
    auto state_result = DeserializeAndValidate(engine_path, inputs, outputs);
    if (!state_result.has_value()) {
        return tl::make_unexpected(state_result.error());
    }

    inputs_ = std::move(inputs);
    outputs_ = std::move(outputs);
    state_.store(std::move(*state_result));
    return {};
}

auto TensorRtEngine::Infer() noexcept -> Result<void> {
    auto state = state_.load();
    if (state == nullptr || state->context == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "TensorRtEngine: no engine loaded",
            .source_location = std::source_location::current(),
        });
    }

    // 校验所有 output binding 均已设置非空 device_ptr。
    for (const auto& output : outputs_) {
        if (output.device_ptr == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Inference_InvalidBinding,
                .message = "TensorRtEngine: output binding '" + output.name +
                           "' has null device_ptr",
                .source_location = std::source_location::current(),
            });
        }
    }

    // TRT 10 enqueueV3(nullptr) 使用默认 CUDA stream。
    if (!state->context->enqueueV3(nullptr)) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "TensorRtEngine: enqueueV3 failed",
            .source_location = std::source_location::current(),
        });
    }

    auto cuda_err = cudaStreamSynchronize(nullptr);
    if (cuda_err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "TensorRtEngine: cudaStreamSynchronize failed: " +
                       std::string(cudaGetErrorString(cuda_err)),
            .source_location = std::source_location::current(),
        });
    }

    return {};
}

auto TensorRtEngine::InferAsync(void* stream) noexcept -> Result<void> {
    auto state = state_.load();
    if (state == nullptr || state->context == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "TensorRtEngine: no engine loaded",
            .source_location = std::source_location::current(),
        });
    }

    // 校验所有 output binding 均已设置非空 device_ptr。
    for (const auto& output : outputs_) {
        if (output.device_ptr == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Inference_InvalidBinding,
                .message = "TensorRtEngine: output binding '" + output.name +
                           "' has null device_ptr",
                .source_location = std::source_location::current(),
            });
        }
    }

    auto* cuda_stream = static_cast<cudaStream_t>(stream);
    if (!state->context->enqueueV3(cuda_stream)) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "TensorRtEngine: enqueueV3 (async) failed",
            .source_location = std::source_location::current(),
        });
    }

    // 不调用 cudaStreamSynchronize——调用方通过 GpuStreamQueue 回调恢复协程。
    return {};
}

auto TensorRtEngine::Reload(const std::filesystem::path& engine_path) noexcept -> Result<void> {
    // 反序列化新 engine + 创建新 context + 校验 binding。
    auto new_state_result = DeserializeAndValidate(engine_path, inputs_, outputs_);
    if (!new_state_result.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_ReloadFailed,
            .message = "TensorRtEngine: Reload failed: " +
                       new_state_result.error().message,
            .source_location = std::source_location::current(),
        });
    }

    // atomic swap——新 context 生效，旧 context 随 shared_ptr 引用计数自然释放。
    state_.store(std::move(*new_state_result));
    return {};
}

auto TensorRtEngine::SetTensorAddress(const std::string& name,
                                        void* device_ptr) noexcept -> Result<void> {
    // 更新 input bindings。
    for (auto& binding : inputs_) {
        if (binding.name == name) {
            binding.device_ptr = device_ptr;
            // 同步更新 TRT execution context（若已加载）。
            auto state = state_.load();
            if (state != nullptr && state->context != nullptr) {
                if (!state->context->setTensorAddress(name.c_str(), device_ptr)) {
                    return tl::make_unexpected(ErrorInfo{
                        .code = ErrorCode::Inference_InvalidBinding,
                        .message = "TensorRtEngine: setTensorAddress failed for '" +
                                   name + "' on TRT context",
                        .source_location = std::source_location::current(),
                    });
                }
            }
            return {};
        }
    }

    // 更新 output bindings。
    for (auto& binding : outputs_) {
        if (binding.name == name) {
            binding.device_ptr = device_ptr;
            auto state = state_.load();
            if (state != nullptr && state->context != nullptr) {
                if (!state->context->setTensorAddress(name.c_str(), device_ptr)) {
                    return tl::make_unexpected(ErrorInfo{
                        .code = ErrorCode::Inference_InvalidBinding,
                        .message = "TensorRtEngine: setTensorAddress failed for '" +
                                   name + "' on TRT context",
                        .source_location = std::source_location::current(),
                    });
                }
            }
            return {};
        }
    }

    return tl::make_unexpected(ErrorInfo{
        .code = ErrorCode::Inference_InvalidBinding,
        .message = "TensorRtEngine: binding '" + name + "' not found",
        .source_location = std::source_location::current(),
    });
}

auto TensorRtEngine::InputBindings() const noexcept -> const std::vector<TensorBinding>& {
    return inputs_;
}

auto TensorRtEngine::OutputBindings() const noexcept -> const std::vector<TensorBinding>& {
    return outputs_;
}

}  // namespace sai::inference
