#include "stage_nodes.h"

#include <chrono>
#include <vector>

#include <sai/pipeline/rule_eval_output.h>
#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/rule/fact_builder.h>

namespace sai::pipeline {

RuleEvalStage::RuleEvalStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    rule_file_ = config["rule_file"].as<std::string>("");
}

auto RuleEvalStage::GetType() const noexcept -> StageType { return StageType::RuleEval; }
auto RuleEvalStage::GetId() const -> std::string_view { return id_; }

auto RuleEvalStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
    // RuleEngine, KnowledgeGraph, VectorPath are concrete classes —
    // set externally via setters before Start().
    if (kg_ && vp_) {
        fact_builder_ = std::make_unique<rule::FactBuilder>(kg_, vp_);
    }
    if (rule_engine_) stub_ = false;
    return {};
}

auto RuleEvalStage::OnStart(Context&) -> Result<void> {
    if (rule_engine_ && !rule_file_.empty()) {
        auto load_result = rule_engine_->LoadFromYAML(rule_file_);
        if (!load_result) return load_result;
        stub_ = false;
    }
    return {};
}
auto RuleEvalStage::OnStop(Context&) -> Result<void> { return {}; }

auto RuleEvalStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        return StageOutput::MakeWithContext(input, std::move(*failure));
    }

    if (auto* det = input.GetIf<sai::detection::DetectionResult>()) {
        rule::FactBase fb;

        if (!stub_ && rule_engine_) {
            // Build FactBase from DetectionResult + Knowledge + Retrieval
            if (fact_builder_) {
                auto surface_id = det->surface_id.empty()
                    ? "default" : det->surface_id;
                auto build_result = fact_builder_->Build(
                    surface_id, *det, {});
                if (!build_result) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = build_result.error().code,
                        .stage_id = id_,
                        .message = build_result.error().message,
                        .surface_id = det->surface_id,
                        .position_id = det->position_id,
                    });
                }
                fb = std::move(*build_result);
            } else {
                // No knowledge/retrieval: populate bare detection facts.
                fb.Set("detection.anomaly_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
                fb.Set("detection.image_level_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
                fb.Set("detection.anomaly_map.max_score",
                    rule::Value::Of(static_cast<double>(det->anomaly_map.MaxScore())),
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
            if (!eval_result) {
                return StageOutput::MakeWithContext(input, PipelineFailure{
                    .code = eval_result.error().code,
                    .stage_id = id_,
                    .message = eval_result.error().message,
                    .surface_id = det->surface_id,
                    .position_id = det->position_id,
                });
            }

            auto resolved = rule_engine_->ResolveConflicts(*eval_result);

            // ── Write GroundTruth record for Bayesian auto-tuning ──
            // Records detection_score so that TuningObjective::EvaluateSimulated
            // can replay candidate thresholds against raw anomaly scores,
            // enabling true Shadow Mode evaluation for the GP surrogate.
            if (recorder_ && recorder_->IsEnabled()) {
                auto surface_id =
                    det->surface_id.empty() ? "default" : det->surface_id;
                std::string machine_verdict = "OK";
                for (const auto& r : resolved) {
                    if (r.matched && r.action.base_severity > 0.0) {
                        machine_verdict = "NG";
                        break;
                    }
                }
                auto now_us = std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
                recorder_->WriteRecord(
                    static_cast<double>(det->image_level_score),
                    machine_verdict, surface_id, now_us);
            }

            RuleEvalOutput output{std::move(fb), std::move(resolved)};
            output.surface_id = det->surface_id;
            output.position_id = det->position_id;
            return StageOutput::MakeWithContext(input, std::move(output));
        }

        return StageOutput::MakeWithContext(input, PipelineFailure{
            .code = sai::ErrorCode::Rule_ParseError,
            .stage_id = id_,
            .message = "RuleEvalStage: rule engine is not configured",
            .surface_id = det->surface_id,
            .position_id = det->position_id,
        });
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "RuleEval expects DetectionResult input"});
}

}  // namespace sai::pipeline
