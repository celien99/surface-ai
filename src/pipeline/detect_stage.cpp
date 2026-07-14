#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }

auto DetectStage::OnInitialize(Context& ctx) -> Result<void> {
    auto detector = ctx.Resolve<detection::IDetector>();
    if (detector) {
        detector_ = *detector;
        stub_ = false;
    }
    return {};
}

auto DetectStage::OnStart(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStop(Context&) -> Result<void> { return {}; }

auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        if (!stub_ && detector_) {
            // IDetector::Detect takes an Embedding, but our pipeline has a
            // DetectionResult from InferenceStage. For MockEngine the
            // DetectionResult is empty; the real flow would embed from
            // patch features. For the stub/mock path, pass-through.
            //
            // Production: InferenceStage would produce embeddings, and this
            // stage would call detector_->Detect(embedding).
            // For now: passthrough the DetectionResult (which may have been
            // populated by InferenceStage's adapter logic).
        }
        return StageOutput(std::move(*det));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects DetectionResult input"});
}

}  // namespace sai::pipeline
