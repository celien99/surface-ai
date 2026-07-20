// tests/reasoner/integration_test.cpp
// End-to-end: FactBase → RuleEngine → DefaultReasoner → ReasoningResult
//
// Replaces the placeholder with a real M5 pipeline integration test.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sai/core/error.h"
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
    auto path = tmp / ("m5_integration_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

// ===========================================================================
// FullPipelineScratchDefect
//
// End-to-end: FactBase → RuleEngine → DefaultReasoner → ReasoningResult
//
// Simulates a scratch defect on a car seat surface (area=12.3mm²,
// position=seat_center). Verifies:
//   - scratch_major rule matches, scratch_minor is overridden
//   - Verdict = "NG" with severity > 0.5
//   - Trace contains all 4 levels (Expression, Rule, TreeBranch, Scoring)
//   - Evidence carries FactSource metadata (Direct + GraphPath)
// ===========================================================================
TEST(M5IntegrationTest, FullPipelineScratchDefect) {
    // -----------------------------------------------------------------------
    // 1. Inline YAML: rule definitions
    //    Two rules: scratch_major (area > 10) overrides scratch_minor (area > 2)
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
    //    Branch on defect.type → scratch → branch on defect.area_mm2
    //      >10 → Leaf NG (formula: weighted score)
    //      default → Leaf WARN
    //    default → Leaf UNCERTAIN
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
    // 3. Build FactBase with simulated detection data (hand-crafted, no real
    //    inference). Includes both Direct mapping entries and a GraphPath
    //    entry with SQL provenance to exercise FactSource tracking.
    // -----------------------------------------------------------------------
    sai::rule::FactBase fb;
    fb.Set("defect.type",    sai::rule::Value::Of(std::string("scratch")),
           {sai::rule::FactSourceKind::Direct, "DetectionResult"});
    fb.Set("defect.area_mm2", sai::rule::Value::Of(12.3),
           {sai::rule::FactSourceKind::Direct, "DetectionResult"});
    fb.Set("defect.position", sai::rule::Value::Of(std::string("seat_center")),
           {sai::rule::FactSourceKind::Direct, "DetectionResult"});
    fb.Set("defect.score", sai::rule::Value::Of(0.92),
           {sai::rule::FactSourceKind::Direct, "PatchCore"});
    fb.Set("defect.confidence", sai::rule::Value::Of(0.85),
           {sai::rule::FactSourceKind::Direct, "PatchCore"});
    fb.Set("material.supplier.batch.reject_rate", sai::rule::Value::Of(0.032),
           {sai::rule::FactSourceKind::GraphPath,
            "material->supplier->batch.reject_rate",
            std::chrono::microseconds(150),
            std::optional<std::string>(
                "SELECT reject_rate FROM batches WHERE id = ?"),
            std::nullopt});

    // -----------------------------------------------------------------------
    // 4. Load rules, evaluate, resolve conflicts
    // -----------------------------------------------------------------------
    sai::rule::RuleEngine engine;
    ASSERT_TRUE(engine.LoadFromYAML(rules_path))
        << "Failed to load rules YAML";

    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    auto resolved = engine.ResolveConflicts(*results);

    // scratch_major should match; scratch_minor should be overridden
    bool major_matched = false;
    bool minor_overridden = false;
    for (const auto& r : resolved) {
        if (r.name == "scratch_major")  major_matched = r.matched;
        if (r.name == "scratch_minor")  minor_overridden = !r.matched;
    }
    EXPECT_TRUE(major_matched)
        << "scratch_major should match (area=12.3 > 10)";
    EXPECT_TRUE(minor_overridden)
        << "scratch_minor should be overridden by scratch_major";

    // -----------------------------------------------------------------------
    // 5. Load decision tree and reason
    // -----------------------------------------------------------------------
    auto tree = sai::reasoner::DecisionTree::LoadFromYAML(tree_path);
    ASSERT_TRUE(tree.has_value()) << tree.error().message;

    sai::reasoner::DefaultReasoner reasoner(std::move(*tree));
    auto result = reasoner.Reason(fb, resolved);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // -----------------------------------------------------------------------
    // 6. Verify ReasoningResult
    // -----------------------------------------------------------------------

    // 6a. Verdict = "NG" (scratch > 10mm² leaf label)
    EXPECT_EQ(result->verdict, "NG");

    // 6b. Severity > 0.5
    //     sigmoid(0.5*0.92 + 0.3*0.85 + 0.2*0.032 - 0.5)
    //     = sigmoid(0.46 + 0.255 + 0.0064 - 0.5)
    //     = sigmoid(0.2214)
    //     ~ 0.555
    EXPECT_GT(result->severity, 0.5);
    EXPECT_GT(result->confidence, 0.0);

    // 6c. Trace has all 4 levels
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
    EXPECT_TRUE(has_expression)
        << "Trace missing Expression level";
    EXPECT_TRUE(has_rule)
        << "Trace missing Rule level";
    EXPECT_TRUE(has_treebranch)
        << "Trace missing TreeBranch level";
    EXPECT_TRUE(has_scoring)
        << "Trace missing Scoring level";

    // 6d. Evidence contains FactSource metadata
    bool found_direct_fact    = false;
    bool found_graphpath_fact = false;
    for (const auto& ev : result->evidence) {
        if (ev.key == "defect.type") {
            found_direct_fact = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::Direct);
            EXPECT_EQ(ev.source.description, "DetectionResult");
        }
        if (ev.key == "material.supplier.batch.reject_rate") {
            found_graphpath_fact = true;
            EXPECT_EQ(ev.source.kind, sai::rule::FactSourceKind::GraphPath);
            EXPECT_TRUE(ev.source.sql.has_value());
            EXPECT_EQ(*ev.source.sql,
                      "SELECT reject_rate FROM batches WHERE id = ?");
        }
    }
    EXPECT_TRUE(found_direct_fact)
        << "Evidence should contain fact 'defect.type'";
    EXPECT_TRUE(found_graphpath_fact)
        << "Evidence should contain fact 'material.supplier.batch.reject_rate'";

    // 6e. Triggered / overridden rule lists
    ASSERT_EQ(result->triggered_rules.size(), 1u);
    EXPECT_EQ(result->triggered_rules[0], "scratch_major");
    ASSERT_EQ(result->overridden_rules.size(), 1u);
    EXPECT_EQ(result->overridden_rules[0], "scratch_minor");

    // Cleanup
    std::filesystem::remove(rules_path);
    std::filesystem::remove(tree_path);
}

}  // namespace
