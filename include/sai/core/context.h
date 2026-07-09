#pragma once

#include <concepts>
#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/lifecycle.h>
#include <sai/core/module.h>
#include <sai/core/registry.h>
#include <sai/core/service.h>
#include <sai/core/type_registry.h>

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

    template <Reflectable T>
        requires std::derived_from<T, IService>
    auto Register(std::shared_ptr<T> instance) -> Result<void>;

    template <Reflectable T>
        requires std::derived_from<T, IService>
    [[nodiscard]] auto Resolve() const -> Result<std::shared_ptr<T>>;

    auto Initialize() -> Result<void>;
    auto Start() -> Result<void>;
    auto Stop() -> Result<void>;

    [[nodiscard]] auto CurrentState() const noexcept -> LifecycleState;

private:
    LifecycleState state_ = LifecycleState::Created;
    std::vector<std::unique_ptr<IModule>> modules_;
    Registry<IService> service_registry_;
};

template <Reflectable T>
    requires std::derived_from<T, IService>
auto Context::Register(std::shared_ptr<T> instance) -> Result<void> {
    if (state_ != LifecycleState::Created) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Lifecycle_RegisterAfterAssembly,
            "cannot register a service after assembly",
            std::source_location::current(),
        });
    }
    return service_registry_.Register(T::kStaticTypeId, std::move(instance));
}

template <Reflectable T>
    requires std::derived_from<T, IService>
auto Context::Resolve() const -> Result<std::shared_ptr<T>> {
    return service_registry_.Resolve(T::kStaticTypeId)
        .map([](const std::shared_ptr<IService>& service) {
            return std::static_pointer_cast<T>(service);
        });
}

}  // namespace sai
