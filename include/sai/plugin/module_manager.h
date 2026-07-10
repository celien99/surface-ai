#pragma once

#include <memory>

#include <sai/core/context.h>
#include <sai/core/error.h>
#include <sai/core/module.h>

namespace sai {

// Registration forwarder for compile-time statically-linked builtin modules.
// Holds no Registry (see 1.3-core-plugin-system.md §3) — only a Context
// reference, forwarding whatever module it receives to
// Context::RegisterModule as-is.
class ModuleManager {
public:
    explicit ModuleManager(Context& context) noexcept;

    ModuleManager(const ModuleManager&) = delete;
    ModuleManager& operator=(const ModuleManager&) = delete;
    ModuleManager(ModuleManager&&) = delete;
    ModuleManager& operator=(ModuleManager&&) = delete;

    // Call order is dependency order (dependees registered first), matching
    // Context::RegisterModule's existing ordering requirement exactly —
    // ModuleManager performs no ordering inference or validation of its own.
    auto RegisterBuiltin(std::unique_ptr<IModule> module) -> Result<void>;

private:
    Context& context_;
};

}  // namespace sai
