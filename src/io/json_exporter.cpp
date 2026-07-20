#include <sai/io/exporter.h>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

namespace sai::io {

auto JsonExporter::Export(const InspectionResult& result,
                          std::filesystem::path output_dir,
                          const sai::image::SurfaceImage* annotated_image) noexcept
    -> Result<void> {
    // 1. Create output directory: output_dir / sku_id / serial_number
    auto export_path = output_dir / result.sku_id / result.serial_number;
    std::error_code ec;
    std::filesystem::create_directories(export_path, ec);
    if (ec) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Io_ExportPathCreateFailed, "Failed to create export directory: " + export_path.string()});
    }

    // 2. Build JSON from InspectionResult
    nlohmann::json j;
    j["sku_id"] = result.sku_id;
    j["serial_number"] = result.serial_number;
    j["timestamp"] = result.timestamp.time_since_epoch().count();
    j["verdict"] = result.verdict;

    auto defects_json = nlohmann::json::array();
    for (const auto& d : result.defects) {
        nlohmann::json defect;
        defect["label"] = d.label;
        defect["severity"] = d.severity;
        defect["confidence"] = d.confidence;
        defect["location"] = {
            {"x", d.location.x},
            {"y", d.location.y},
            {"width", d.location.width},
            {"height", d.location.height},
        };
        defect["evidence_path"] = d.evidence_path;
        defects_json.push_back(std::move(defect));
    }
    j["defects"] = std::move(defects_json);

    // 3. Write result.json
    auto json_path = export_path / "result.json";
    std::ofstream ofs(json_path);
    if (!ofs) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Io_SerializationFailed, "Failed to open file for writing: " + json_path.string()});
    }
    ofs << j.dump(2);
    if (!ofs) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Io_SerializationFailed, "Failed to write JSON to: " + json_path.string()});
    }

    // 4. If annotated image provided, write PPM placeholder
    // TODO: real PNG when a PNG lib is added.
    if (annotated_image != nullptr) {
        auto ppm_path = export_path / "annotated.ppm";
        std::ofstream ppm(ppm_path, std::ios::binary);
        if (!ppm) {
            return tl::make_unexpected(ErrorInfo{ErrorCode::Io_SerializationFailed, "Failed to open PPM file for writing: " + ppm_path.string()});
        }

        const auto& meta = annotated_image->Meta();
        const auto* pixels = annotated_image->Data();
        auto size_bytes = annotated_image->SizeBytes();

        // P6 PPM header: P6\n<width> <height>\n<maxval>\n
        ppm << "P6\n" << meta.width << " " << meta.height << "\n255\n";
        ppm.write(reinterpret_cast<const char*>(pixels),
                  static_cast<std::streamsize>(size_bytes));
        if (!ppm) {
            return tl::make_unexpected(ErrorInfo{ErrorCode::Io_SerializationFailed, "Failed to write PPM data to: " + ppm_path.string()});
        }
    }

    return {};
}

}  // namespace sai::io
