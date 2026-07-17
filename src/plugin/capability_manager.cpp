#include <mutex>

#include <sai/plugin/capability_manager.h>

namespace sai {

auto CapabilityManager::RegisterKnownCapability(std::string capability) -> Result<void> {
    std::unique_lock lock(mutex_);
    auto [it, inserted] = known_capabilities_.insert(std::move(capability));
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "capability already registered",
            std::source_location::current(),
        });
    }
    return {};
}

auto CapabilityManager::Validate(const std::vector<std::string>& declared_capabilities) const
    -> Result<void> {
    std::shared_lock lock(mutex_);
    for (const auto& capability : declared_capabilities) {
        if (!known_capabilities_.contains(capability)) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Plugin_CapabilityUnsupported,
                "declared capability '" + capability + "' is not registered as known",
                std::source_location::current(),
            });
        }
    }
    return {};
}

}  // namespace sai
