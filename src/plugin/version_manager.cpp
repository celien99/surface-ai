#include <sai/plugin/version_manager.h>

namespace sai {

namespace {

auto Compare(const SemVer& lhs, const SemVer& rhs) noexcept -> int {
    if (lhs.major != rhs.major) {
        return lhs.major < rhs.major ? -1 : 1;
    }
    if (lhs.minor != rhs.minor) {
        return lhs.minor < rhs.minor ? -1 : 1;
    }
    if (lhs.patch != rhs.patch) {
        return lhs.patch < rhs.patch ? -1 : 1;
    }
    return 0;
}

}  // namespace

auto VersionManager::CheckCompatible(const VersionRange& required, const SemVer& actual) noexcept
    -> Result<void> {
    bool const below_min = Compare(actual, required.min_inclusive) < 0;
    bool const at_or_above_max = Compare(actual, required.max_exclusive) >= 0;
    if (below_min || at_or_above_max) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Plugin_VersionIncompatible,
            "actual version outside required [min_inclusive, max_exclusive) range",
            std::source_location::current(),
        });
    }
    return {};
}

}  // namespace sai
