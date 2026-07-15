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

auto RuleEvalStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
    // RuleEngine, KnowledgeGraph, VectorPath are concrete classes, not
    // IService subtypes — set externally via setters before Start().
    if (!rule_file_.empty() && rule_engine_) {
        auto load_result = rule_engine_->LoadFromYAML(rule_file_);
        if (!load_result) return load_result;
    }
    if (kg_ && vp_) {
        fact_builder_ = std::make_unique<rule::FactBuilder>(kg_, vp_);
    }
    if (rule_engine_) stub_ = false;
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

            // --- Vector retrieval via CLIP global features ---
            if (fact_builder_ && !det->global_features.empty()) {
                // Store CLIP features in FactBase as a Value list
                std::vector<rule::Value> feature_vals;
                feature_vals.reserve(det->global_features.size());
                for (auto f : det->global_features) {
                    feature_vals.push_back(
                        rule::Value::Of(static_cast<double>(f)));
                }
                fb.Set("embedding.global",
                       rule::Value::OfList(std::move(feature_vals)),
                       {rule::FactSourceKind::Direct, "CLIP GlobalEmbedder"});

                // Trigger FAISS vector search for similar historical cases
                auto retrieval_result =
                    fact_builder_->RunVectorRetrieval(fb, "embedding.global");
                if (!retrieval_result) {
                    // Retrieval failure is non-fatal: rule eval continues
                    // with detection facts only.
                    (void)retrieval_result.error();
                }
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
