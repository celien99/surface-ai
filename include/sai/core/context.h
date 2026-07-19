#pragma once

#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/lifecycle.h>
#include <sai/core/module.h>

namespace sai {

class Context {
public:
    Context() noexcept = default;
    ~Context() = default;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    auto RegisterModule(std::unique_ptr<IModule> module) -> Result<void>;

    auto Initialize() -> Result<void>;
    auto Start() -> Result<void>;
    auto Stop() -> Result<void>;

    [[nodiscard]] auto CurrentState() const noexcept -> LifecycleState;

private:
    LifecycleState state_ = LifecycleState::Created;
    std::vector<std::unique_ptr<IModule>> modules_;
};

}  // namespace sai
