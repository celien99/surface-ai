#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>
#include <sai/embedding/embedding.h>

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
    if (auto* emb = std::get_if<sai::embedding::Embedding>(&input)) {
        sai::detection::DetectionResult result;
        if (!stub_ && detector_) {
            auto det_result = detector_->Detect(*emb);
            if (det_result) result = std::move(*det_result);
        }
        // Carry forward CLIP global features for RuleEvalStage.
        if (emb->HasGlobalFeatures()) {
            result.global_features = emb->GlobalFeatures();
        }
        // Carry forward surface identity.
        if (!emb->SurfaceId().empty()) {
            result.surface_id = emb->SurfaceId();
        }
        return StageOutput(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}

}  // namespace sai::pipeline
