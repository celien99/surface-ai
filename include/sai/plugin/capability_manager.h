#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <sai/core/error.h>

namespace sai {

// Tracks the set of capability tags known to this deployment, registered by
// the application entry point during the assembly phase (e.g. "camera.gige",
// "detector.onnx"). Validate checks every capability tag a plugin declares
// against that set during PluginManager::Load, catching typo'd tags that
// would otherwise be silently accepted as "valid but never recognized by
// anyone". See 1.3-core-plugin-system.md §4.
class CapabilityManager {
public:
    // Duplicate registration of the same tag returns Core_TypeAlreadyRegistered
    // rather than overwriting, following the same "do not overwrite" decision
    // path as 1.1 batch's TypeRegistry::Register (see §4). This batch's own
    // §12 Future Extension explicitly locks its Plugin_* contribution to the
    // four codes already listed in error.h (VersionIncompatible/
    // CapabilityUnsupported/LicenseInvalid/CircularDependency) and defers the
    // rest of the taxonomy to batch 1.6 — minting a fifth Plugin_* code here
    // would exceed that locked scope, so this reuses Core_TypeAlreadyRegistered
    // literally instead of introducing a new code.
    auto RegisterKnownCapability(std::string capability) -> Result<void>;

    // Returns Plugin_CapabilityUnsupported when any tag in
    // `declared_capabilities` is not present in the known set.
    [[nodiscard]] auto Validate(const std::vector<std::string>& declared_capabilities) const
        -> Result<void>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_set<std::string> known_capabilities_;
};

}  // namespace sai
