#pragma once

#include <cstdint>
#include <source_location>
#include <string>

#include <tl/expected.hpp>

namespace sai {

enum class ErrorCode : std::uint32_t {
    Core_Unknown = 0,
    Core_ConstructionFailed,
    Core_TypeAlreadyRegistered,
    Core_TypeNotFound,
    Lifecycle_RegisterAfterAssembly,
    Memory_ArenaExhausted,
    Memory_RequestExceedsSlabSize,
    Memory_PoolExhausted,
    Plugin_VersionIncompatible,
    Plugin_CapabilityUnsupported,
    Plugin_LicenseInvalid,
    Plugin_CircularDependency,
    Runtime_QueueFull,
    Runtime_Cancelled,
    Runtime_NodeNotFound,
    Infra_LogSinkInitFailed,
    Infra_ConfigFileNotFound,       // ConfigStore::Load 指定路径不存在
    Infra_ConfigParseError,         // yaml-cpp 解析阶段失败（YAML 语法错误）
    Infra_ConfigValidationFailed,   // ConfigSchema::Validate 未通过
    Infra_ConfigKeyNotFound,        // ConfigStore::Get<T> 查询的键不存在
    Infra_ConfigKeyTypeMismatch,    // 键存在但无法转换为请求的 T
    Device_ConnectionFailed,
    Device_NotConnected,
    Device_AcquisitionInProgress,
    Image_UnsupportedPixelFormat,
    Image_DimensionMismatch,
    Image_PreprocessFailed,
};

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::source_location source_location;
};

template <typename T>
using Result = tl::expected<T, ErrorInfo>;

}  // namespace sai
