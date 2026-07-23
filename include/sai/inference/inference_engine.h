// inference_engine.h — 推理引擎抽象接口（批次 3.1）
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>

namespace sai::inference {

enum class TensorDataType : std::uint8_t {
    Unknown = 0,
    Float16,
    Float32,
    Int8,
    Int32,
    Bool,
};

[[nodiscard]] constexpr auto TensorDataTypeSize(TensorDataType dtype) noexcept
    -> std::size_t {
    switch (dtype) {
        case TensorDataType::Float16: return 2;
        case TensorDataType::Float32: return 4;
        case TensorDataType::Int8: return 1;
        case TensorDataType::Int32: return 4;
        case TensorDataType::Bool: return 1;
        case TensorDataType::Unknown: return 0;
    }
    return 0;
}

struct TensorBinding {
    std::string name;
    std::vector<std::int64_t> shape;
    std::size_t size_bytes = 0;
    void* device_ptr = nullptr;
    TensorDataType dtype = TensorDataType::Unknown;

    [[nodiscard]] auto ElementCount() const noexcept -> std::size_t {
        if (shape.empty()) return 0;
        std::size_t count = 1;
        for (auto extent : shape) {
            if (extent <= 0) return 0;
            count *= static_cast<std::size_t>(extent);
        }
        return count;
    }

    [[nodiscard]] auto ExpectedSizeBytes() const noexcept -> std::size_t {
        return ElementCount() * TensorDataTypeSize(dtype);
    }
};

class IInferenceEngine : public Object {
public:
    [[nodiscard]] virtual auto Load(const std::filesystem::path& engine_path,
                                     std::vector<TensorBinding> inputs,
                                     std::vector<TensorBinding> outputs) noexcept
        -> Result<void> = 0;

    [[nodiscard]] virtual auto Infer() noexcept -> Result<void> = 0;
    // stream 在目标平台为 cudaStream_t，可移植头文件使用 void*。
    [[nodiscard]] virtual auto InferAsync(void* stream) noexcept -> Result<void> = 0;

    [[nodiscard]] virtual auto Reload(const std::filesystem::path& engine_path) noexcept
        -> Result<void> = 0;

    // 更新指定 I/O tensor 的设备端地址——adapter 每帧可能从 GpuPool 分配/复用不同 slab，
    // TensorRT 需要知道当前帧的数据位置。name 在 Load() 的 bindings 中首次注册，此处仅更新地址。
    [[nodiscard]] virtual auto SetTensorAddress(const std::string& name,
                                                  void* device_ptr) noexcept -> Result<void> = 0;

    [[nodiscard]] virtual auto InputBindings() const noexcept
        -> const std::vector<TensorBinding>& = 0;
    [[nodiscard]] virtual auto OutputBindings() const noexcept
        -> const std::vector<TensorBinding>& = 0;
};

}  // namespace sai::inference
