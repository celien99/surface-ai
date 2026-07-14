#include "stage_nodes.h"

#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>
#include <sai/reasoner/decision_tree.h>

namespace sai::pipeline {

ReasonStage::ReasonStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    tree_file_ = config["tree_file"].as<std::string>("");
}

auto ReasonStage::GetType() const noexcept -> StageType { return StageType::Reason; }
auto ReasonStage::GetId() const -> std::string_view { return id_; }

auto ReasonStage::OnInitialize(Context& ctx) -> Result<void> {
    auto reasoner = ctx.Resolve<reasoner::IReasoner>();
    if (reasoner) {
        reasoner_ = *reasoner;
        stub_ = false;
    }
    return {};
}

auto ReasonStage::OnStart(Context&) -> Result<void> { return {}; }
auto ReasonStage::OnStop(Context&) -> Result<void> { return {}; }

auto ReasonStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* eval_output = std::get_if<RuleEvalOutput>(&input)) {
        if (!stub_ && reasoner_) {
            auto result = reasoner_->Reason(eval_output->facts, eval_output->rules);
            if (!result) return tl::make_unexpected(result.error());
            return StageOutput(std::move(*result));
        }
        // Stub: return empty ReasoningResult
        return StageOutput(sai::reasoner::ReasoningResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Reason expects RuleEvalOutput input"});
}

}  // namespace sai::pipeline
