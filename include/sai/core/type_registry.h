#pragma once

#include <concepts>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include <sai/core/error.h>
#include <sai/core/reflectable.h>
#include <sai/core/type_id.h>

namespace sai {

struct TypeInfo {
    TypeId id;
    std::string_view name;
};

template <typename T>
concept Reflectable = std::is_base_of_v<IReflectable, T> &&
    requires {
        { T::kStaticTypeId } -> std::convertible_to<TypeId>;
        { T::kStaticTypeName } -> std::convertible_to<std::string_view>;
    };

class TypeRegistry {
public:
    static auto Instance() noexcept -> TypeRegistry&;

    template <Reflectable T>
    auto Register() -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<TypeInfo>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<TypeId, TypeInfo> entries_;
};

template <Reflectable T>
auto TypeRegistry::Register() -> Result<void> {
    std::unique_lock lock(mutex_);
    auto [it, inserted] = entries_.try_emplace(
        T::kStaticTypeId, TypeInfo{T::kStaticTypeId, T::kStaticTypeName});
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "type already registered",
            std::source_location::current(),
        });
    }
    return {};
}

}  // namespace sai
