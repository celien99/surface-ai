// exporter.h — 导出插件接口
#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/service.h>
#include <sai/core/rect.h>
#include <sai/image/surface_image.h>
#include <sai/plugin/plugin.h>

namespace sai::io {

using sai::core::Rect;

struct DefectRecord {
    std::string label;            // "划痕", "破洞", "褶皱"
    std::string severity;         // "CRITICAL" | "MAJOR" | "MINOR"
    float confidence = 0.0F;
    Rect location;
    std::string evidence_path;   // 裁剪缺陷子图路径（可选）
};

struct InspectionResult {
    std::string sku_id;          // "Tesla_Model3_Driver"
    std::string serial_number;
    std::chrono::system_clock::time_point timestamp;
    std::vector<DefectRecord> defects;
    std::string verdict;         // "PASS" | "FAIL" | "RECHECK"
};

class IExporter : public IPlugin, public IService {
public:
    SAI_DECLARE_TYPE_ID(sai.io.exporter)

    [[nodiscard]] virtual auto Export(const InspectionResult& result,
                                      std::filesystem::path output_dir,
                                      const sai::image::SurfaceImage* annotated_image) noexcept
        -> Result<void> = 0;
    [[nodiscard]] virtual auto FormatName() const noexcept -> std::string_view = 0;
};

// JsonExporter 是内置导出器，编译进 sai_io 静态库，不走 dlopen。
// 无状态，所有生命周期方法为 no-op。
class JsonExporter final : public IExporter {
public:
    SAI_DECLARE_TYPE_ID(sai.io.json-exporter)

    [[nodiscard]] auto Export(const InspectionResult& result,
                              std::filesystem::path output_dir,
                              const sai::image::SurfaceImage* annotated_image) noexcept
        -> Result<void> override;

    [[nodiscard]] auto FormatName() const noexcept -> std::string_view override
    { return "json_report"; }

    // Lifecycle（无状态导出器，直接通过）
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

private:
    PluginManifest manifest_{};
};

}  // namespace sai::io
