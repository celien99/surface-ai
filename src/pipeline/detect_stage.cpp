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
        if (!stub_ && detector_) {
            auto result = detector_->Detect(*emb);
            if (result) return StageOutput(std::move(*result));
            // On failure, fall through to empty DetectionResult
        }
        return StageOutput(sai::detection::DetectionResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}

}  // namespace sai::pipeline
