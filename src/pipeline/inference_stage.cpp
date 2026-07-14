#include "stage_nodes.h"
#include <sai/detection/detection_result.h>

namespace sai::pipeline {

InferenceStage::InferenceStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto InferenceStage::GetType() const noexcept -> StageType { return StageType::Inference; }
auto InferenceStage::GetId() const -> std::string_view { return id_; }
auto InferenceStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto InferenceStage::OnStart(Context&) -> Result<void> { return {}; }
auto InferenceStage::OnStop(Context&) -> Result<void> { return {}; }

auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    // Mock: SurfaceImage -> DetectionResult
    if (auto* img = std::get_if<sai::image::SurfaceImage>(&input)) {
        return StageOutput(sai::detection::DetectionResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
