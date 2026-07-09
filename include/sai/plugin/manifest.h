#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai {

struct SemVer {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
};

// [min_inclusive, max_exclusive) half-open interval; max_exclusive is
// normally the next major version that would introduce a breaking change.
// See 1.3-core-plugin-system.md §3/§4.
struct VersionRange {
    SemVer min_inclusive;
    SemVer max_exclusive;
};

struct PluginDependency {
    std::string plugin_name;  // Matches the depended-on plugin's manifest name field.
    VersionRange required_version;
};

// In-memory representation of a deserialized plugin.yaml. Only the fields
// are frozen by the design doc; the YAML text layout is an implementation
// detail (see §2 Responsibilities).
struct PluginManifest {
    std::string name;  // Fully-qualified name; hashed via detail::Fnv1aHash to
                        // produce this plugin's TypeId, must match the string
                        // passed to SAI_DECLARE_TYPE_ID in the plugin's code.
    std::string library_path;  // Path to the .so, relative to the manifest's directory.
    SemVer version;
    std::vector<std::string> capabilities;
    std::vector<PluginDependency> dependencies;
    std::string license_token;  // Raw credential handed to LicenseManager::Validate.
};

// Reads and deserializes a single plugin.yaml file at `manifest_path` into a
// PluginManifest. This is a pure, read-only step: it never touches the
// paired .so (no dlopen) and never runs any version/capability/license
// validation (see §3 Design — validation happens before dlopen, not during
// manifest parsing). Not part of the design doc's frozen §4 signatures; the
// entry point itself is a reasonable, in-scope addition since the design doc
// explicitly leaves the deserialization entry point unlocked (see §2).
[[nodiscard]] auto LoadManifest(const std::filesystem::path& manifest_path)
    -> Result<PluginManifest>;

}  // namespace sai
