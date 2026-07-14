#include "stage_nodes.h"
#include <sai/detection/detection_result.h>

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }
auto DetectStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStart(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStop(Context&) -> Result<void> { return {}; }

auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    // Passthrough: DetectionResult -> DetectionResult
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        return StageOutput(std::move(*det));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects DetectionResult input"});
}

}  // namespace sai::pipeline
