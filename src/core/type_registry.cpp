#include <sai/core/type_registry.h>

namespace sai {

auto TypeRegistry::Instance() noexcept -> TypeRegistry& {
    static TypeRegistry instance;
    return instance;
}

auto TypeRegistry::Resolve(TypeId id) const -> Result<TypeInfo> {
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
