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
    // The full cross-module error code taxonomy is completed by batch 1.6,
    // out of scope for this scaffold.
};

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::source_location source_location;
};

template <typename T>
using Result = tl::expected<T, ErrorInfo>;

}  // namespace sai
