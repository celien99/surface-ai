#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include <sai/core/error.h>
#include <sai/core/type_id.h>

namespace sai {

template <typename TInterface>
class Registry {
public:
    auto Register(TypeId id, std::shared_ptr<TInterface> instance) -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<std::shared_ptr<TInterface>>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<TypeId, std::shared_ptr<TInterface>> entries_;
};

template <typename TInterface>
auto Registry<TInterface>::Register(TypeId id, std::shared_ptr<TInterface> instance)
    -> Result<void> {
    std::unique_lock lock(mutex_);
    auto [it, inserted] = entries_.try_emplace(id, std::move(instance));
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "type already registered",
            std::source_location::current(),
        });
    }
    return {};
}

template <typename TInterface>
auto Registry<TInterface>::Resolve(TypeId id) const -> Result<std::shared_ptr<TInterface>> {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeNotFound,
            "type not found",
            std::source_location::current(),
        });
    }
    return it->second;
}

}  // namespace sai
