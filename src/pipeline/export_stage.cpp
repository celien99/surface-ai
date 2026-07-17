#include "stage_nodes.h"

#include <optional>

#include <sai/image/surface_image.h>
#include <sai/pipeline/pipeline.h>
#include <sai/reasoner/reasoner.h>
#include <sai/io/exporter.h>

namespace sai::pipeline {

ExportStage::ExportStage(std::string id, YAML::Node config, Pipeline* pipeline)
    : id_(std::move(id)), pipeline_(pipeline) {
    auto out_dir = config["output_dir"].as<std::string>("/tmp/surface-ai/results/");
    output_dir_ = std::filesystem::path(out_dir);
}

auto ExportStage::GetType() const noexcept -> StageType { return StageType::Export; }
auto ExportStage::GetId() const -> std::string_view { return id_; }

auto ExportStage::OnInitialize(Context& ctx) -> Result<void> {
    auto exporter = ctx.Resolve<io::IExporter>();
    if (exporter) {
        exporter_ = *exporter;
        stub_ = false;
    }
    return {};
}

auto ExportStage::OnStart(Context&) -> Result<void> { return {}; }
auto ExportStage::OnStop(Context&) -> Result<void> { return {}; }

auto ExportStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* result = std::get_if<sai::reasoner::ReasoningResult>(&input)) {
        if (!stub_ && exporter_) {
            // Build InspectionResult from ReasoningResult
            io::InspectionResult inspection;
            inspection.sku_id = "default";
            inspection.serial_number = "unknown";
            inspection.timestamp = std::chrono::system_clock::now();
            inspection.verdict = result->verdict;

            // Map triggered rules to DefectRecords
            for (const auto& rule_name : result->triggered_rules) {
                io::DefectRecord defect;
                defect.label = rule_name;
                defect.severity = result->verdict;
                defect.confidence = static_cast<float>(result->confidence);
                inspection.defects.push_back(std::move(defect));
            }

            // Create output dir
            std::filesystem::create_directories(output_dir_);

            // Retrieve the per-frame image pixel snapshot from the pipeline
            // side channel (stored by Preprocess stage worker).
            // Reconstruct as SurfaceImage for the exporter.
            auto snapshot = pipeline_ ? pipeline_->TakeFrameImage()
                                       : std::nullopt;
            std::optional<sai::image::SurfaceImage> frame_image;
            if (snapshot.has_value()) {
                frame_image = sai::image::SurfaceImage::FromOwnedBuffer(
                    std::move(snapshot->first), snapshot->second);
            }

            auto export_result = exporter_->Export(
                inspection, output_dir_,
                frame_image.has_value() ? &*frame_image : nullptr);
            if (!export_result) return tl::make_unexpected(export_result.error());
        }
        return StageOutput(std::move(*result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Export expects ReasoningResult input"});
}

}  // namespace sai::pipeline
