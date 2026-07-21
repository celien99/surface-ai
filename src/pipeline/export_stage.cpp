#include "stage_nodes.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <vector>

#include <sai/image/surface_image.h>
#include <sai/reasoner/reasoner.h>
#include <sai/io/exporter.h>

namespace sai::pipeline {

namespace {

auto FrameSerial(const StageInput& input, std::uint16_t position_id) -> std::string {
    auto serial = "pos_" + std::to_string(position_id);
    if (input.Frame()) {
        serial += "_frame_" + std::to_string(input.Frame()->frame_id);
    }
    return serial;
}

auto DefectSeverity(double score) -> std::string {
    if (score >= 0.8) return "CRITICAL";
    if (score >= 0.5) return "MAJOR";
    return "MINOR";
}

// Write a heatmap PPM from anomaly scores (grid_h × grid_w → upsampled to w×h).
// Color map: blue (cold) → green → yellow → red (hot).
auto WriteHeatmapPpm(const std::filesystem::path& path,
                     const std::vector<float>& scores,
                     std::size_t grid_h, std::size_t grid_w,
                     std::size_t out_w, std::size_t out_h) -> void {
    if (scores.empty() || out_w == 0 || out_h == 0) return;

    // Simple nearest-neighbor upsample: grid position → output pixel.
    std::vector<std::uint8_t> rgb(out_w * out_h * 3);
    float min_score = *std::min_element(scores.begin(), scores.end());
    float max_score = *std::max_element(scores.begin(), scores.end());
    float range = max_score - min_score;
    if (range <= 0.0F) range = 1.0F;

    for (std::size_t y = 0; y < out_h; ++y) {
        for (std::size_t x = 0; x < out_w; ++x) {
            auto gy = static_cast<std::size_t>(
                static_cast<float>(y) * static_cast<float>(grid_h) / static_cast<float>(out_h));
            auto gx = static_cast<std::size_t>(
                static_cast<float>(x) * static_cast<float>(grid_w) / static_cast<float>(out_w));
            if (gy >= grid_h) gy = grid_h - 1;
            if (gx >= grid_w) gx = grid_w - 1;

            float t = (scores[gy * grid_w + gx] - min_score) / range;
            if (t < 0.0F) t = 0.0F;
            if (t > 1.0F) t = 1.0F;

            // Jet-like color map: blue(0) → cyan → green → yellow → red(1)
            std::uint8_t r, g, b;
            if (t < 0.25F) {
                b = 255;
                g = static_cast<std::uint8_t>(t / 0.25F * 255.0F);
                r = 0;
            } else if (t < 0.5F) {
                b = static_cast<std::uint8_t>(255 - (t - 0.25F) / 0.25F * 255.0F);
                g = 255;
                r = 0;
            } else if (t < 0.75F) {
                b = 0;
                g = 255;
                r = static_cast<std::uint8_t>((t - 0.5F) / 0.25F * 255.0F);
            } else {
                b = 0;
                g = static_cast<std::uint8_t>(255 - (t - 0.75F) / 0.25F * 255.0F);
                r = 255;
            }
            auto idx = (y * out_w + x) * 3;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }

    std::ofstream ppm(path, std::ios::binary);
    if (!ppm) return;
    ppm << "P6\n" << out_w << " " << out_h << "\n255\n";
    ppm.write(reinterpret_cast<const char*>(rgb.data()),
              static_cast<std::streamsize>(rgb.size()));
}

}  // namespace

ExportStage::ExportStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    auto out_dir = config["output_dir"].as<std::string>("/tmp/surface-ai/results/");
    output_dir_ = std::filesystem::path(out_dir);
}

auto ExportStage::GetType() const noexcept -> StageType { return StageType::Export; }
auto ExportStage::GetId() const -> std::string_view { return id_; }

auto ExportStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
    // IExporter is injected via SetExporter() before Start().
    if (exporter_) stub_ = false;
    return {};
}

auto ExportStage::OnStart(Context&) -> Result<void> { return {}; }
auto ExportStage::OnStop(Context&) -> Result<void> { return {}; }

auto ExportStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        sai::reasoner::ReasoningResult result;
        result.surface_id = failure->surface_id;
        result.position_id = failure->position_id;
        result.verdict = "RECHECK";
        result.recommendation = failure->stage_id + ": " + failure->message;
        result.confidence = 0.0;

        if (!stub_ && exporter_) {
            io::InspectionResult inspection;
            inspection.sku_id = result.surface_id.empty() ? "default" : result.surface_id;
            inspection.serial_number = FrameSerial(input, result.position_id);
            inspection.timestamp = std::chrono::system_clock::now();
            inspection.verdict = result.verdict;
            auto export_result = exporter_->Export(inspection, output_dir_, nullptr);
            if (!export_result) return tl::make_unexpected(export_result.error());
        }
        return StageOutput::MakeWithContext(input, std::move(result));
    }

    if (auto* result = input.GetIf<sai::reasoner::ReasoningResult>()) {
        if (stub_ || !exporter_) {
            result->verdict = "RECHECK";
            result->recommendation = "ExportStage: exporter is not configured";
            return StageOutput::MakeWithContext(input, std::move(*result));
        }

        if (!stub_ && exporter_) {
            // Build InspectionResult from ReasoningResult
            io::InspectionResult inspection;
            inspection.sku_id = result->surface_id.empty() ? "default" : result->surface_id;
            inspection.serial_number = FrameSerial(input, result->position_id);
            inspection.timestamp = std::chrono::system_clock::now();
            inspection.verdict = result->verdict;

            // Map triggered rules to DefectRecords
            for (const auto& rule_name : result->triggered_rules) {
                io::DefectRecord defect;
                defect.label = rule_name;
                defect.severity = DefectSeverity(result->severity);
                defect.confidence = static_cast<float>(result->confidence);
                inspection.defects.push_back(std::move(defect));
            }

            // Create output dir
            std::filesystem::create_directories(output_dir_);

            // Reconstruct the frame image captured in the shared frame context.
            std::optional<sai::image::SurfaceImage> frame_image;
            if (input.Frame() && input.Frame()->image.has_value()) {
                auto& snapshot = *input.Frame()->image;
                frame_image = sai::image::SurfaceImage::FromOwnedBuffer(
                    std::move(snapshot.first), snapshot.second);
            }

            auto export_result = exporter_->Export(
                inspection, output_dir_,
                frame_image.has_value() ? &*frame_image : nullptr);
            if (!export_result) return tl::make_unexpected(export_result.error());

            // Write anomaly heatmap if available
            if (input.Frame() && input.Frame()->anomaly.has_value()
                && !input.Frame()->anomaly->scores.empty()) {
                auto& anomaly = *input.Frame()->anomaly;
                auto heatmap_path = output_dir_ / inspection.sku_id
                    / inspection.serial_number / "heatmap.ppm";
                WriteHeatmapPpm(heatmap_path, anomaly.scores,
                                anomaly.grid_h, anomaly.grid_w,
                                518, 518);
            }
        }
        return StageOutput::MakeWithContext(input, std::move(*result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Export expects ReasoningResult input"});
}

}  // namespace sai::pipeline
