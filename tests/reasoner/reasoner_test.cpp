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

namespace sai::reasoner {
namespace {

// ===========================================================================
// Helpers
// ===========================================================================

/// Write a YAML string to a temporary file and return its path.
auto TempYAMLPath(const std::string& content) -> std::filesystem::path {
    static std::size_t counter = 0;
    auto tmp = std::filesystem::temp_directory_path();
    auto path =
        tmp / ("reasoner_test_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

/// Build a DecisionTree from inline YAML.
auto MakeTree(const std::string& yaml) -> std::unique_ptr<DecisionTree> {
    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    EXPECT_TRUE(result.has_value()) << result.error().message;
    std::filesystem::remove(path);
    return std::move(*result);
}

// ===========================================================================
// Reason_SimpleTree
// ===========================================================================
TEST(ReasonerTest, Reason_SimpleTree) {
    // Tree: branch on "defect_type" → scratch→NG, dirt→WARN, default→OK
    // Each leaf has formulas evaluated against fact fields.
    auto yaml = R"(
type: branch
field: defect_type
branches:
  scratch:
    type: leaf
    label: NG
    recommendation: inspect_scratch_region
    formulas:
      - weights: [1.0]
        features: [scratch_score]
        threshold: 0.0
  dirt:
    type: leaf
    label: WARN
    recommendation: clean_surface
    formulas:
      - weights: [1.0]
        features: [dirt_score]
        threshold: 0.0
default:
  type: leaf
  label: OK
  recommendation: continue
)";
    auto tree = MakeTree(yaml);
    DefaultReasoner reasoner(std::move(tree));

    rule::FactBase facts;
    facts.Set("defect_type", rule::Value::Of(std::string("scratch")),
              {rule::FactSourceKind::Direct, "camera_feed"});
    facts.Set("scratch_score", rule::Value::Of(0.85),
              {rule::FactSourceKind::Direct, "model_scorer"});

    auto result = reasoner.Reason(facts, {});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->verdict, "NG");
    EXPECT_EQ(result->recommendation, "inspect_scratch_region");
    EXPECT_GT(result->confidence, 0.0);
    EXPECT_TRUE(result->triggered_rules.empty());
    EXPECT_TRUE(result->overridden_rules.empty());

    // Trace should have at least a TreeBranch + Scoring entry.
    ASSERT_GE(result->trace.size(), 2);
    EXPECT_EQ(result->trace[0].level, rule::TraceStep::Level::TreeBranch);
    EXPECT_EQ(result->trace[1].level, rule::TraceStep::Level::Scoring);
}

// ===========================================================================
// Reason_DefaultBranch
// ===========================================================================
TEST(ReasonerTest, Reason_DefaultBranch) {
    // Same tree; "unknown" has no matching branch → default with label
    // UNCERTAIN.
    auto yaml = R"(
type: branch
field: defect_type
branches:
  scratch:
    type: leaf
    label: NG
    recommendation: inspect
default:
  type: leaf
  label: UNCERTAIN
  recommendation: manual_review
)";
    auto tree = MakeTree(yaml);
    DefaultReasoner reasoner(std::move(tree));

    rule::FactBase facts;
    facts.Set("defect_type", rule::Value::Of(std::string("unknown")),
              {rule::FactSourceKind::Direct, "test"});

    auto result = reasoner.Reason(facts, {});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->verdict, "UNCERTAIN");
    EXPECT_EQ(result->recommendation, "manual_review");
}

// ===========================================================================
// Reason_TraceContainsAllLevels
// ===========================================================================
TEST(ReasonerTest, Reason_TraceContainsAllLevels) {
    auto yaml = R"(
type: branch
field: defect_type
branches:
  scratch:
    type: leaf
    label: NG
    recommendation: inspect
default:
  type: leaf
  label: OK
  recommendation: pass
)";
    auto tree = MakeTree(yaml);
    DefaultReasoner reasoner(std::move(tree));

    rule::FactBase facts;
    facts.Set("defect_type", rule::Value::Of(std::string("scratch")),
              {rule::FactSourceKind::Direct, "test"});

    // Simulate a resolved rule with an Expression-level eval_trace.
    rule::ResolvedRule rule;
    rule.name = "depth_rule";
    rule.matched = true;
    rule.action = {.label = "NG", .base_severity = 0.8, .recommendation = "check"};
    rule.eval_trace.push_back({
        "expr_1",
        rule::TraceStep::Level::Expression,
        "depth > 0.5",
        "depth_rule.yaml:10",
        std::chrono::microseconds(100),
        std::nullopt,
    });

    auto result = reasoner.Reason(facts, {rule});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    bool has_expression = false;
    bool has_rule = false;
    bool has_treebranch = false;
    bool has_scoring = false;

    for (auto& step : result->trace) {
        switch (step.level) {
        case rule::TraceStep::Level::Expression:
            has_expression = true;
            break;
        case rule::TraceStep::Level::Rule:
            has_rule = true;
            break;
        case rule::TraceStep::Level::TreeBranch:
            has_treebranch = true;
            break;
        case rule::TraceStep::Level::Scoring:
            has_scoring = true;
            break;
        }
    }

    EXPECT_TRUE(has_expression) << "Trace missing Expression level";
    EXPECT_TRUE(has_rule) << "Trace missing Rule level";
    EXPECT_TRUE(has_treebranch) << "Trace missing TreeBranch level";
    EXPECT_TRUE(has_scoring) << "Trace missing Scoring level";
}

// ===========================================================================
// Reason_EvidenceContainsFactSources
// ===========================================================================
TEST(ReasonerTest, Reason_EvidenceContainsFactSources) {
    auto yaml = R"(
type: branch
field: defect_type
branches:
  scratch:
    type: leaf
    label: NG
    recommendation: inspect
default:
  type: leaf
  label: OK
  recommendation: pass
)";
    auto tree = MakeTree(yaml);
    DefaultReasoner reasoner(std::move(tree));

    rule::FactBase facts;
    facts.Set("defect_type", rule::Value::Of(std::string("scratch")),
              {rule::FactSourceKind::Direct, "camera_feed",
               std::chrono::microseconds(50), std::nullopt, std::nullopt});

    // Resolved rule to verify it also appears in evidence.
    rule::ResolvedRule rule;
    rule.name = "depth_rule";
    rule.matched = true;
    rule.action = {.label = "NG", .base_severity = 0.8, .recommendation = "check"};
    rule.eval_trace.push_back({
        "expr_1",
        rule::TraceStep::Level::Expression,
        "depth > 0.5",
        "depth_rule.yaml:10",
        std::chrono::microseconds(100),
        std::nullopt,
    });

    auto result = reasoner.Reason(facts, {rule});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Verify evidence contains the fact entry with its source metadata.
    bool found_defect_fact = false;
    bool found_depth_rule = false;
    for (auto& e : result->evidence) {
        if (e.key == "defect_type") {
            found_defect_fact = true;
            EXPECT_EQ(e.source.kind, rule::FactSourceKind::Direct);
            EXPECT_EQ(e.source.description, "camera_feed");
        }
        if (e.key == "depth_rule") {
            found_depth_rule = true;
        }
    }

    EXPECT_TRUE(found_defect_fact)
        << "Evidence should contain fact entry 'defect_type'";
    EXPECT_TRUE(found_depth_rule)
        << "Evidence should contain rule entry 'depth_rule'";
}

// ===========================================================================
// Reason_MissingFieldGoesToDefault
// ===========================================================================
TEST(ReasonerTest, Reason_MissingFieldGoesToDefault) {
    // Tree branches on "defect_type"; facts lack this key → default branch.
    auto yaml = R"(
type: branch
field: defect_type
branches:
  scratch:
    type: leaf
    label: NG
    recommendation: inspect
default:
  type: leaf
  label: UNCERTAIN
  recommendation: manual_review
)";
    auto tree = MakeTree(yaml);
    DefaultReasoner reasoner(std::move(tree));

    rule::FactBase facts;
    // "defect_type" is intentionally absent.
    facts.Set("other_field", rule::Value::Of(42.0),
              {rule::FactSourceKind::Direct, "test"});

    auto result = reasoner.Reason(facts, {});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->verdict, "UNCERTAIN");
    EXPECT_EQ(result->recommendation, "manual_review");
}

}  // namespace
}  // namespace sai::reasoner
