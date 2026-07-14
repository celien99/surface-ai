#include "stage_nodes.h"
#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

ReasonStage::ReasonStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto ReasonStage::GetType() const noexcept -> StageType { return StageType::Reason; }
auto ReasonStage::GetId() const -> std::string_view { return id_; }
auto ReasonStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto ReasonStage::OnStart(Context&) -> Result<void> { return {}; }
auto ReasonStage::OnStop(Context&) -> Result<void> { return {}; }

auto ReasonStage::Process(StageInput input) -> Result<StageOutput> {
    // Mock: ResolvedRule[] -> ReasoningResult
    if (auto* rules = std::get_if<std::vector<sai::rule::ResolvedRule>>(&input)) {
        return StageOutput(sai::reasoner::ReasoningResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Reason expects ResolvedRule[] input"});
}

}  // namespace sai::pipeline
