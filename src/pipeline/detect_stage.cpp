#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }

auto DetectStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
    return {};
}

auto DetectStage::OnStart(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStop(Context&) -> Result<void> { return {}; }

auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        // IDetector::Detect(Embedding) requires an Embedding, but the
        // current pipeline produces DetectionResult from InferenceStage.
        // Once an Embedding stage is inserted between Inference and Detect,
        // this will call detector_->Detect(embedding) here.
        (void)detector_;  // wired, ready for future Embedding stage
        return StageOutput(std::move(*det));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects DetectionResult input"});
}

}  // namespace sai::pipeline
