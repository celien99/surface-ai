// rule_reasoner_pipeline_test.cpp — Task 14: M5 端到端集成测试
// 全链路: Rule YAML → RuleEngine → DecisionTree YAML → DefaultReasoner → ReasoningResult
//
// 验证跨 sai::rule + sai::reasoner + sai::knowledge + sai::retrieval + sai::detection
// 模块集成的完整推理管线。

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sai/detection/detection_result.h"
#include "sai/reasoner/decision_tree.h"
#include "sai/reasoner/evidence_collector.h"
#include "sai/reasoner/reasoner.h"
#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"
#include "sai/rule/value.h"

namespace {

// ===========================================================================
// Helper — write content to a temporary YAML file and return its path.
// ===========================================================================
auto TempYAML(const std::string& content) -> std::filesystem::path {
    static std::size_t counter = 0;
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / ("m5_cross_module_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

// ===========================================================================
// M5Pipeline_RuleReasonerEndToEnd
//
// Cross-module pipeline: YAML rules → EvaluateAll → ResolveConflicts →
// DecisionTree → DefaultReasoner → ReasoningResult.
//
// Uses rule engine for conflict resolution (overrides) and the reasoner for
// decision tree traversal and weighted scoring. Excludes live KnowledgeGraph /
// VectorPath; FactBase is hand-crafted.
// ===========================================================================
TEST(M5PipelineTest, RuleReasonerEndToEnd) {
    // -----------------------------------------------------------------------
    // 1. Inline YAML: rule definitions — scratch defect overrides
    // -----------------------------------------------------------------------
    auto rules_yaml = R"(
rule_sets:
  seat_leather_inspection:
    - name: "scratch_major"
      priority: 100
      condition:
        and:
          - field: defect.type
            op: eq
            value: "scratch"
          - field: defect.area_mm2
            op: gt
            value: 10
      action:
        label: "NG"
        base_severity: 0.8
        recommendation: "Severe scratch requiring rework"
      overrides: ["scratch_minor"]
      overridden_by: []
    - name: "scratch_minor"
      priority: 50
      condition:
        and:
          - field: defect.type
            op: eq
            value: "scratch"
          - field: defect.area_mm2
            op: gt
            value: 2
      action:
        label: "WARN"
        base_severity: 0.3
        recommendation: "Minor scratch, track"
      overrides: []
      overridden_by: ["scratch_major"]
)";

    // -----------------------------------------------------------------------
    // 2. Inline YAML: decision tree
    // -----------------------------------------------------------------------
    auto tree_yaml = R"(
type: branch
field: defect.type
branches:
  scratch:
    type: branch
    field: defect.area_mm2
    branches:
      ">10":
        type: leaf
        label: NG
        recommendation: "Severe scratch detected"
        formulas:
          - weights: [0.5, 0.3, 0.2]
            features: [defect.score, defect.confidence, material.supplier.batch.reject_rate]
            threshold: 0.5
    default:
      type: leaf
      label: WARN
      recommendation: "Minor scratch suspected"
      formulas:
        - weights: [1.0]
          features: [defect.score]
          threshold: 0.3
default:
  type: leaf
  label: UNCERTAIN
  recommendation: "Unknown defect type"
  formulas:
    - weights: [1.0]
      features: [defect.score]
      threshold: 0.5
)";

    auto rules_path = TempYAML(rules_yaml);
    auto tree_path  = TempYAML(tree_yaml);

    // -----------------------------------------------------------------------
    // 3. Build FactBase with simulated detection data
    // -----------------------------------------------------------------------
    sai::rule::FactBase fb;
    fb.Set("defect.type",    sai::rule::Value::Of(std::string("scratch")),
           {sai::rule::FactSourceKind::Direct, "DetectionResult::type"});
    fb.Set("defect.area_mm2", sai::rule::Value::Of(12.3),
           {sai::rule::FactSourceKind::Direct, "DetectionResult::area_mm2"});
    fb.Set("defect.score", sai::rule::Value::Of(0.92),
           {sai::rule::FactSourceKind::Direct, "PatchCore::image_level_score"});
    fb.Set("defect.confidence", sai::rule::Value::Of(0.85),
           {sai::rule::FactSourceKind::Direct, "PatchCore::confidence"});
    fb.Set("material.supplier.batch.reject_rate", sai::rule::Value::Of(0.032),
           {sai::rule::FactSourceKind::GraphPath,
            "material->supplier->batch.reject_rate",
            std::chrono::microseconds(150),
            std::optional<std::string>(
                "SELECT reject_rate FROM batches WHERE id = ?"),
            std::nullopt});

    // -----------------------------------------------------------------------
    // 4. Rule pipeline: load → evaluate → resolve conflicts
    // -----------------------------------------------------------------------
    sai::rule::RuleEngine engine;
    ASSERT_TRUE(engine.LoadFromYAML(rules_path))
        << "Failed to load rules YAML";

    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    auto resolved = engine.ResolveConflicts(*results);

    bool major_matched = false;
    bool minor_overridden = false;
    for (const auto& r : resolved) {
        if (r.name == "scratch_major")  major_matched = r.matched;
        if (r.name == "scratch_minor")  minor_overridden = !r.matched;
    }
    EXPECT_TRUE(major_matched);
    EXPECT_TRUE(minor_overridden);

    // -----------------------------------------------------------------------
    // 5. Reasoner pipeline: decision tree → default reasoner
    // -----------------------------------------------------------------------
    auto tree = sai::reasoner::DecisionTree::LoadFromYAML(tree_path);
    ASSERT_TRUE(tree.has_value()) << tree.error().message;

    sai::reasoner::DefaultReasoner reasoner(std::move(*tree));
    auto result = reasoner.Reason(fb, resolved);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // -----------------------------------------------------------------------
    // 6. Verify ReasoningResult
    // -----------------------------------------------------------------------

    // 6a. Verdict
    EXPECT_EQ(result->verdict, "NG");

    // 6b. Severity > 0.5
    EXPECT_GT(result->severity, 0.5);
    EXPECT_GT(result->confidence, 0.0);

    // 6c. Recommendation from leaf
    EXPECT_EQ(result->recommendation, "Severe scratch detected");

    // 6d. Trace has all 4 levels
    bool has_expression   = false;
    bool has_rule         = false;
    bool has_treebranch   = false;
    bool has_scoring      = false;
    for (const auto& step : result->trace) {
        switch (step.level) {
        case sai::rule::TraceStep::Level::Expression:
            has_expression = true;
            break;
        case sai::rule::TraceStep::Level::Rule:
            has_rule = true;
            break;
        case sai::rule::TraceStep::Level::TreeBranch:
            has_treebranch = true;
            break;
        case sai::rule::TraceStep::Level::Scoring:
            has_scoring = true;
            break;
        }
    }
    EXPECT_TRUE(has_expression);
    EXPECT_TRUE(has_rule);
    EXPECT_TRUE(has_treebranch);
    EXPECT_TRUE(has_scoring);

    // 6e. Trace has reasonable number of steps
    EXPECT_GE(result->trace.size(), 5u)
        << "Expected at least 5 trace steps (1 Expression + 1 Rule "
           "+ 2 TreeBranch + 1 Scoring)";

    // 6f. Evidence contains FactSource metadata
    bool found_defect_type = false;
    bool found_reject_rate = false;
    for (const auto& ev : result->evidence) {
        if (ev.key == "defect.type") {
            found_defect_type = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::Direct);
            EXPECT_EQ(ev.source.description, "DetectionResult::type");
        }
        if (ev.key == "material.supplier.batch.reject_rate") {
            found_reject_rate = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::GraphPath);
            EXPECT_TRUE(ev.source.sql.has_value());
        }
    }
    EXPECT_TRUE(found_defect_type);
    EXPECT_TRUE(found_reject_rate);

    // 6g. Contains rule evidence items
    bool found_rule_major = false;
    bool found_rule_minor = false;
    for (const auto& ev : result->evidence) {
        if (ev.key == "scratch_major") {
            found_rule_major = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::Computed);
        }
        if (ev.key == "scratch_minor") {
            found_rule_minor = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::Default);
        }
    }
    EXPECT_TRUE(found_rule_major);
    EXPECT_TRUE(found_rule_minor);

    // 6h. Triggered / overridden rule lists
    ASSERT_EQ(result->triggered_rules.size(), 1u);
    EXPECT_EQ(result->triggered_rules[0], "scratch_major");
    ASSERT_EQ(result->overridden_rules.size(), 1u);
    EXPECT_EQ(result->overridden_rules[0], "scratch_minor");

    std::filesystem::remove(rules_path);
    std::filesystem::remove(tree_path);
}

// ===========================================================================
// M5Pipeline_DetectionResultHandCrafted
//
// Verifies that a hand-crafted DetectionResult (M3) can be used as input.
// While FactBuilder is not exercised here (requires live KnowledgeGraph),
// the test shows that DetectionResult fields conceptually feed into FactBase.
// ===========================================================================
TEST(M5PipelineTest, DetectionResultHandCrafted) {
    // Construct a minimal DetectionResult as M3 would produce it.
    sai::detection::AnomalyMap amap;
    amap.scores = {0.92f};
    amap.grid_h = 1;
    amap.grid_w = 1;

    sai::detection::RegionProposal region;
    region.bounding_box = {0, 0, 50, 50};
    region.max_anomaly_score = 0.92f;
    region.mean_anomaly_score = 0.85f;
    region.area_pixels = 2500;

    sai::detection::DetectionResult det;
    det.anomaly_map = std::move(amap);
    det.regions.push_back(std::move(region));
    det.image_level_score = 0.92f;
    det.inference_latency = std::chrono::nanoseconds(42'000'000);

    // Simulate FactBuilder's mapping of DetectionResult → FactBase.
    sai::rule::FactBase fb;
    fb.Set("defect.type",    sai::rule::Value::Of(std::string("scratch")),
           {sai::rule::FactSourceKind::Direct, "DetectionResult::type"});
    fb.Set("defect.score",   sai::rule::Value::Of(det.image_level_score),
           {sai::rule::FactSourceKind::Direct, "DetectionResult::image_level_score"});
    fb.Set("defect.area_mm2", sai::rule::Value::Of(12.3),
           {sai::rule::FactSourceKind::Direct, "DetectionResult::region"});
    fb.Set("defect.confidence", sai::rule::Value::Of(
               static_cast<double>(det.regions[0].mean_anomaly_score)),
           {sai::rule::FactSourceKind::Computed,
            "mean anomaly score of top region"});

    // Verify FactBase correctly reflects DetectionResult data
    auto type_val = fb.Get("defect.type");
    ASSERT_TRUE(type_val.has_value());
    auto type_str = type_val->AsString();
    ASSERT_TRUE(type_str.has_value());
    EXPECT_EQ(*type_str, "scratch");

    auto score_val = fb.Get("defect.score");
    ASSERT_TRUE(score_val.has_value());
    auto score_d = score_val->AsDouble();
    ASSERT_TRUE(score_d.has_value());
    // float → double conversion may introduce small rounding error
    EXPECT_NEAR(*score_d, 0.92, 1e-6);

    auto conf_val = fb.Get("defect.confidence");
    ASSERT_TRUE(conf_val.has_value());
    auto conf_d = conf_val->AsDouble();
    ASSERT_TRUE(conf_d.has_value());
    EXPECT_NEAR(*conf_d, 0.85, 1e-6);

    // Verify source provenance
    auto& type_source = fb.SourceOf("defect.type");
    EXPECT_EQ(type_source.kind, sai::rule::FactSourceKind::Direct);
    EXPECT_EQ(type_source.description, "DetectionResult::type");

    auto& conf_source = fb.SourceOf("defect.confidence");
    EXPECT_EQ(conf_source.kind, sai::rule::FactSourceKind::Computed);

    // AllEntries should contain 4 entries
    EXPECT_EQ(fb.AllEntries().size(), 4u);
}

}  // namespace
