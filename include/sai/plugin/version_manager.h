#pragma once

#include <sai/core/error.h>
#include <sai/plugin/manifest.h>

namespace sai {

// Stateless validator: holds no record of any previously-checked version,
// each call judges only the two arguments passed in. See
// 1.3-core-plugin-system.md §4.
class VersionManager {
public:
    // Returns Plugin_VersionIncompatible when `actual` falls outside
    // [required.min_inclusive, required.max_exclusive).
    [[nodiscard]] static auto CheckCompatible(const VersionRange& required,
                                               const SemVer& actual) noexcept -> Result<void>;
};

}  // namespace sai
