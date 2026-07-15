// importer.h — 导入插件接口
#pragma once

#include <filesystem>
#include <memory>
#include <string_view>

#include <sai/core/error.h>
#include <sai/image/image.h>
#include <sai/plugin/plugin.h>

#include <yaml-cpp/yaml.h>

namespace sai::io {

using sai::image::Image;

// DatasetEntry: a single image in a multi-light dataset manifest.
// ImportDataset() returns entries; caller imports pixels via ImportImage()
// and sets metadata on each RawImage.
struct DatasetEntry {
    std::filesystem::path path;
    std::string surface_id;
    std::uint16_t position_id = 0;
    std::uint16_t light_id = 0;
};

class IImporter : public IPlugin {
public:
    [[nodiscard]] virtual auto ImportImage(std::filesystem::path path) noexcept
        -> Result<std::unique_ptr<Image>> = 0;
    [[nodiscard]] virtual auto ImportMetadata(std::filesystem::path path) noexcept
        -> Result<YAML::Node> = 0;
    [[nodiscard]] virtual auto FormatName() const noexcept -> std::string_view = 0;
};

// BasicImporter 是内置导入器，编译进 sai_io 静态库，不走 dlopen。
// 支持 YAML 元数据导入和 P6 PPM 图像导入。无状态，所有生命周期方法为 no-op。
class BasicImporter final : public IImporter {
public:
    SAI_DECLARE_TYPE_ID(sai.io.basic-importer)

    [[nodiscard]] auto ImportImage(std::filesystem::path path) noexcept
        -> Result<std::unique_ptr<Image>> override;
    [[nodiscard]] auto ImportMetadata(std::filesystem::path path) noexcept
        -> Result<YAML::Node> override;

    // ImportDataset reads a YAML manifest and returns structured entries.
    // Does NOT load pixel data — caller iterates and calls ImportImage().
    [[nodiscard]] static auto ImportDataset(std::filesystem::path yaml_path) noexcept
        -> Result<std::vector<DatasetEntry>>;

    [[nodiscard]] auto FormatName() const noexcept -> std::string_view override
    { return "basic_import"; }

    // Lifecycle（无状态导入器，直接通过）
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

private:
    PluginManifest manifest_{};
};

}  // namespace sai::io
