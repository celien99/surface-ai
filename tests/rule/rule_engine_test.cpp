#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sai/rule/rule_engine.h"
#include "sai/rule/value.h"

namespace sai::rule {
namespace {

// ===========================================================================
// Helper — write a string to a temp file and return the path
// ===========================================================================
auto WriteTempYaml(const std::string& content) -> std::filesystem::path {
    auto dir = std::filesystem::temp_directory_path();
    auto path = dir / "sai_rule_test_XXXXXX.yaml";
    // mkstemps needs a modifiable buffer; use a random suffix
    auto tmp = std::filesystem::temp_directory_path() /
               ("sai_rule_test_" + std::to_string(std::rand()) + ".yaml");
    std::ofstream ofs(tmp);
    ofs << content;
    ofs.close();
    return tmp;
}

// ===========================================================================
// LoadFromYAML
// ===========================================================================

TEST(RuleEngineTest, LoadFromYAML) {
    // Use the embedded test fixture
    RuleEngine engine;
    auto path = std::filesystem::path(TEST_DATA_DIR) / "test_rules.yaml";

    auto result = engine.LoadFromYAML(path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Verify rules are loaded via DetectOverlaps (which iterates rule_sets_)
    auto overlaps = engine.DetectOverlaps();
    // overlap_test has 2 rules both referencing "temperature" → 1 warning
    ASSERT_GE(overlaps.size(), 1u);
    bool found = false;
    for (const auto& ow : overlaps) {
        if (ow.rule_a == "high_temp_check" && ow.rule_b == "low_temp_check") {
            found = true;
            EXPECT_EQ(ow.common_fields.size(), 1u);
            EXPECT_EQ(ow.common_fields[0], "temperature");
        }
    }
    EXPECT_TRUE(found) << "expected overlap between high_temp_check and low_temp_check";
}

TEST(RuleEngineTest, LoadFromYAML_MissingRuleSets) {
    RuleEngine engine;
    auto path = WriteTempYaml("other_key: []\n");
    auto result = engine.LoadFromYAML(path);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_ParseError);
    std::filesystem::remove(path);
}

TEST(RuleEngineTest, LoadFromYAML_EmptyName) {
    RuleEngine engine;
    auto path = WriteTempYaml(R"(
rule_sets:
  test_set:
    - priority: 1
      condition: "x > 1"
      action:
        label: "A"
)");
    auto result = engine.LoadFromYAML(path);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_ParseError);
    std::filesystem::remove(path);
}

TEST(RuleEngineTest, LoadFromYAML_InvalidCondition) {
    RuleEngine engine;
    auto path = WriteTempYaml(R"(
rule_sets:
  test_set:
    - name: "bad_rule"
      condition: "x @@@ y"
      action:
        label: "A"
)");
    auto result = engine.LoadFromYAML(path);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_ParseError);
    std::filesystem::remove(path);
}

// ===========================================================================
// EvaluateAll
// ===========================================================================

TEST(RuleEngineTest, EvaluateAll) {
    RuleEngine engine;
    auto path = std::filesystem::path(TEST_DATA_DIR) / "test_rules.yaml";
    ASSERT_TRUE(engine.LoadFromYAML(path));

    FactBase fb;
    fb.Set("temperature", Value::Of(90.0),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("warpage", Value::Of(1.0),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("scratch_count", Value::Of(3.0),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("scratch_depth", Value::Of(0.2),
           FactSource{FactSourceKind::Direct, "test"});

    auto result = engine.EvaluateAll(fb);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto& rules = *result;
    EXPECT_EQ(rules.size(), 5u);  // total across all rule_sets

    // Count matched rules
    int matched_count = 0;
    for (const auto& r : rules) {
        if (r.matched) ++matched_count;
    }
    // temperature=90 → high_temp_reject (t>85) and high_temp_check (t>80) match
    // warpage=1 → warpage_warning (w>2.5) no match
    // scratch_count=3, scratch_depth=0.2 → scratch_reject (c>5 and d>0.5) no match
    // low_temp_check (t<30) no match
    EXPECT_EQ(matched_count, 2);

    // Verify trace steps are populated
    for (const auto& r : rules) {
        if (r.matched) {
            EXPECT_FALSE(r.eval_trace.empty());
        }
    }
}

TEST(RuleEngineTest, EvaluateAll_NoRules) {
    RuleEngine engine;
    FactBase fb;
    auto result = engine.EvaluateAll(fb);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

// ===========================================================================
// EvaluateAll_ParallelRuleSets — multiple rule_sets produce correct aggregate
// ===========================================================================

TEST(RuleEngineTest, EvaluateAll_ParallelRuleSets) {
    RuleEngine engine;
    auto path = std::filesystem::path(TEST_DATA_DIR) / "test_rules.yaml";
    ASSERT_TRUE(engine.LoadFromYAML(path));

    FactBase fb;
    fb.Set("temperature", Value::Of(25.0),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("warpage", Value::Of(0.5),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("scratch_count", Value::Of(0.0),
           FactSource{FactSourceKind::Direct, "test"});
    fb.Set("scratch_depth", Value::Of(0.0),
           FactSource{FactSourceKind::Direct, "test"});

    auto result = engine.EvaluateAll(fb);
    ASSERT_TRUE(result.has_value());
    // All 5 rules from 3 rule_sets should be present
    EXPECT_EQ(result->size(), 5u);
}

// ===========================================================================
// EvaluateAll_SingleRuleFailureSkips
// ===========================================================================

TEST(RuleEngineTest, EvaluateAll_SingleRuleFailureSkips) {
    RuleEngine engine;
    auto path = WriteTempYaml(R"(
rule_sets:
  test_set:
    - name: "good_rule"
      condition: "x > 0"
      action:
        label: "OK"
    - name: "bad_rule"
      condition: "nonexistent > 1.0"
      action:
        label: "ERROR"
    - name: "another_good"
      condition: "x < 10"
      action:
        label: "OK"
)");
    ASSERT_TRUE(engine.LoadFromYAML(path));

    FactBase fb;
    fb.Set("x", Value::Of(5.0),
           FactSource{FactSourceKind::Direct, "test"});

    auto result = engine.EvaluateAll(fb);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // All 3 rules produce a ResolvedRule (no crash)
    EXPECT_EQ(result->size(), 3u);

    // good_rule and another_good should match; bad_rule should not match
    for (const auto& r : *result) {
        if (r.name == "good_rule" || r.name == "another_good") {
            EXPECT_TRUE(r.matched) << r.name << " should have matched";
        } else if (r.name == "bad_rule") {
            EXPECT_FALSE(r.matched) << "bad_rule should have been skipped";
        }
    }

    std::filesystem::remove(path);
}

// ===========================================================================
// EnableHotReload — no-op on non-Linux, just verify it doesn't error
// ===========================================================================

TEST(RuleEngineTest, EnableHotReload) {
    RuleEngine engine;
    std::stop_source source;
    auto result = engine.EnableHotReload("/tmp/nonexistent.yaml",
                                          source.get_token());
    // Always succeeds on non-Linux (no-op)
    EXPECT_TRUE(result.has_value());
}

// ===========================================================================
// DetectOverlaps
// ===========================================================================

TEST(RuleEngineTest, DetectOverlaps) {
    RuleEngine engine;
    auto path = std::filesystem::path(TEST_DATA_DIR) / "test_rules.yaml";
    ASSERT_TRUE(engine.LoadFromYAML(path));

    auto warnings = engine.DetectOverlaps();
    ASSERT_FALSE(warnings.empty());

    // high_temp_check + low_temp_check both reference "temperature"
    bool found = false;
    for (const auto& w : warnings) {
        if (w.rule_a == "high_temp_check" && w.rule_b == "low_temp_check") {
            found = true;
            EXPECT_EQ(w.common_fields.size(), 1u);
            EXPECT_EQ(w.common_fields[0], "temperature");
        }
    }
    EXPECT_TRUE(found);
}

TEST(RuleEngineTest, DetectOverlaps_NoOverlaps) {
    RuleEngine engine;
    auto path = WriteTempYaml(R"(
rule_sets:
  set_a:
    - name: "rule_x"
      condition: "width > 100"
      action:
        label: "A"
    - name: "rule_y"
      condition: "height > 200"
      action:
        label: "B"
)");
    ASSERT_TRUE(engine.LoadFromYAML(path));

    auto warnings = engine.DetectOverlaps();
    EXPECT_TRUE(warnings.empty());

    std::filesystem::remove(path);
}

// ===========================================================================
// ResolveConflicts — basic sanity (nominal case, no overrides)
// ===========================================================================

TEST(RuleEngineTest, ResolveConflicts_Nominal) {
    RuleEngine engine;
    auto path = WriteTempYaml(R"(
rule_sets:
  checks:
    - name: "rule_a"
      condition: "x > 5"
      action:
        label: "A"
      overrides: []
      overridden_by: []
    - name: "rule_b"
      condition: "x > 3"
      action:
        label: "B"
      overrides: []
      overridden_by: []
)");
    ASSERT_TRUE(engine.LoadFromYAML(path));

    FactBase fb;
    fb.Set("x", Value::Of(10.0),
           FactSource{FactSourceKind::Direct, "test"});

    auto evaluated = engine.EvaluateAll(fb);
    ASSERT_TRUE(evaluated);

    // Both match
    EXPECT_EQ(evaluated->size(), 2u);

    // No overrides → both survive
    auto resolved = engine.ResolveConflicts(*evaluated);
    EXPECT_EQ(resolved.size(), 2u);
    int matched = 0;
    for (const auto& r : resolved) {
        if (r.matched) ++matched;
    }
    EXPECT_EQ(matched, 2);

    std::filesystem::remove(path);
}

}  // namespace
}  // namespace sai::rule
