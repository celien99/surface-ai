#include "stage_nodes.h"
#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>

namespace sai::pipeline {

RuleEvalStage::RuleEvalStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto RuleEvalStage::GetType() const noexcept -> StageType { return StageType::RuleEval; }
auto RuleEvalStage::GetId() const -> std::string_view { return id_; }
auto RuleEvalStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto RuleEvalStage::OnStart(Context&) -> Result<void> { return {}; }
auto RuleEvalStage::OnStop(Context&) -> Result<void> { return {}; }

auto RuleEvalStage::Process(StageInput input) -> Result<StageOutput> {
    // Mock: DetectionResult -> ResolvedRule[]
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        return StageOutput(std::vector<sai::rule::ResolvedRule>{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "RuleEval expects DetectionResult input"});
}

}  // namespace sai::pipeline
