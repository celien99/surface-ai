#include <sai/core/context.h>

namespace sai {

auto Context::RegisterModule(std::unique_ptr<IModule> module) -> Result<void> {
    if (state_ != LifecycleState::Created) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Lifecycle_RegisterAfterAssembly,
            "cannot register a module after assembly",
            std::source_location::current(),
        });
    }
    modules_.push_back(std::move(module));
    return {};
}

auto Context::Initialize() -> Result<void> {
    for (auto& module : modules_) {
        auto result = module->OnInitialize(*this);
        if (!result) {
            return result;
        }
    }
    state_ = LifecycleState::Initialized;
    return {};
}

auto Context::Start() -> Result<void> {
    for (auto& module : modules_) {
        auto result = module->OnStart(*this);
        if (!result) {
            return result;
        }
    }
    state_ = LifecycleState::Running;
    return {};
}

auto Context::Stop() -> Result<void> {
    Result<void> first_error{};
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
        auto result = (*it)->OnStop(*this);
        if (!result && first_error) {
            first_error = std::move(result);
        }
    }
    state_ = LifecycleState::Stopped;
    return first_error;
}

auto Context::CurrentState() const noexcept -> LifecycleState {
    return state_;
}

}  // namespace sai
