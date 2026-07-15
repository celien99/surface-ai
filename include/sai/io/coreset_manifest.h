// coreset_manifest.h — Coreset file registry for multi-position detection
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai::io {

struct CoresetBankEntry {
    std::uint16_t position_id = 0;
    std::filesystem::path path;  // relative to manifest file's directory
};

struct CoresetManifest {
    std::string surface_id;
    std::vector<CoresetBankEntry> banks;
};

// Load from YAML file. All bank paths are resolved relative to the
// manifest file's parent directory.
[[nodiscard]] auto LoadCoresetManifest(const std::filesystem::path& yaml_path) noexcept
    -> Result<CoresetManifest>;

// Write manifest to YAML file.
[[nodiscard]] auto SaveCoresetManifest(const std::filesystem::path& yaml_path,
                                        const CoresetManifest& manifest) noexcept
    -> Result<void>;

}  // namespace sai::io
