#include "stage_nodes.h"

#include <filesystem>

#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/rule/fact_builder.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>

namespace sai::pipeline {

RuleEvalStage::RuleEvalStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    rule_file_ = config["rule_file"].as<std::string>("");
}

auto RuleEvalStage::GetType() const noexcept -> StageType { return StageType::RuleEval; }
auto RuleEvalStage::GetId() const -> std::string_view { return id_; }

auto RuleEvalStage::OnInitialize(Context& ctx) -> Result<void> {
    // Resolve RuleEngine
    auto re = ctx.Resolve<rule::RuleEngine>();
    if (!re) return {};  // stay stub
    rule_engine_ = *re;

    // Load YAML rules
    if (!rule_file_.empty()) {
        auto load_result = rule_engine_->LoadFromYAML(rule_file_);
        if (!load_result) return load_result;
    }

    // Try to resolve KnowledgeGraph + VectorPath for FactBuilder
    auto kg = ctx.Resolve<knowledge::KnowledgeGraph>();
    auto vp = ctx.Resolve<retrieval::VectorPath>();
    if (kg && vp) {
        kg_ = *kg;
        vp_ = *vp;
        fact_builder_ = std::make_unique<rule::FactBuilder>(kg_, vp_);
    }

    stub_ = false;
    return {};
}

auto RuleEvalStage::OnStart(Context&) -> Result<void> { return {}; }
auto RuleEvalStage::OnStop(Context&) -> Result<void> { return {}; }

auto RuleEvalStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        rule::FactBase fb;

        if (!stub_ && rule_engine_) {
            // Build FactBase from DetectionResult + Knowledge + Retrieval
            if (fact_builder_) {
                std::vector<std::string> graph_paths;
                // Resolve graph paths specified in rule conditions
                auto all_entries = fb.AllEntries();
                auto build_result = fact_builder_->Build(
                    "default", *det, graph_paths);
                if (build_result) {
                    fb = std::move(*build_result);
                }
                // else: fb stays empty, EvaluateAll handles empty facts
            } else {
                // No knowledge/retrieval: populate bare detection facts
                fb.Set("detection.anomaly_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
                fb.Set("detection.image_level_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
            }

            // Evaluate rules
            auto eval_result = rule_engine_->EvaluateAll(fb);
            if (!eval_result) return tl::make_unexpected(eval_result.error());

            auto resolved = rule_engine_->ResolveConflicts(*eval_result);

            return StageOutput(RuleEvalOutput{std::move(fb), std::move(resolved)});
        }

        // Stub: return empty RuleEvalOutput
        return StageOutput(RuleEvalOutput{std::move(fb), {}});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "RuleEval expects DetectionResult input"});
}

}  // namespace sai::pipeline
