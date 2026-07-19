#include "stage_nodes.h"

#include <sai/pipeline/rule_eval_output.h>
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
    if (auto* eval_output = input.GetIf<RuleEvalOutput>()) {
        sai::reasoner::ReasoningResult result;

        if (!stub_ && reasoner_) {
            auto reason_result = reasoner_->Reason(
                eval_output->facts, eval_output->rules);
            if (!reason_result) return tl::make_unexpected(reason_result.error());
            result = std::move(*reason_result);
        }

        // SAM2 region refinement: if segmenter is available and detection
        // result contains anomaly regions, refine their boundaries.
        // M5 will extend this with point/box prompt types and spatial
        // reasoning that consumes the refined masks.
        if (!stub_ && sam2_segmenter_) {
            // The DetectionResult data is in eval_output->facts as
            // individual scalar fields; the original GpuImage is not
            // accessible at this stage in the current linear pipeline.
            //
            // M5 will add a per-frame side channel (PipelineContext) that
            // carries the original image alongside the DetectionResult so
            // SAM2 can run.  For now the segmenter is wired and callable
            // but deferred to M5 for full activation.
            (void)sam2_segmenter_;  // reserved for M5 spatial reasoning
        }

        return StageOutput::Make(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Reason expects RuleEvalOutput input"});
}

}  // namespace sai::pipeline
