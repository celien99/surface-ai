#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "sai/rule/rule_engine.h"
#include "sai/rule/value.h"
#include "sai/core/error.h"

namespace sai::rule {
namespace {

// ===========================================================================
// Helper — write a string to a temp file and return the path
// ===========================================================================
auto WriteTempYaml(const std::string& content) -> std::filesystem::path {
    auto tmp = std::filesystem::temp_directory_path() /
               ("sai_rule_integration_" + std::to_string(std::rand()) + ".yaml");
    std::ofstream ofs(tmp);
    ofs << content;
    ofs.close();
    return tmp;
}

// ===========================================================================
// EndToEndRuleEvaluation
//
// Full pipeline: FactBase → EvaluateAll → ResolveConflicts
// Verifies:
//   - scratch_major matches and survives conflict resolution
//   - scratch_minor is overridden by scratch_major (matched set to false)
// ===========================================================================
TEST(RuleIntegrationTest, EndToEndRuleEvaluation) {
    // 1. Write YAML with two rules where scratch_major overrides scratch_minor
    auto yaml = R"(
rule_sets:
  scratch_inspection:
    - name: "scratch_major"
      priority: 100
      condition: 'defect.type == "scratch" AND defect.area_mm2 > 10'
      action:
        label: "scratch_major"
        base_severity: 0.8
        recommendation: "严重划痕，建议立即返工"
      overrides: ["scratch_minor"]
      overridden_by: []
    - name: "scratch_minor"
      priority: 50
      condition: 'defect.type == "scratch" AND defect.area_mm2 > 2'
      action:
        label: "scratch_minor"
        base_severity: 0.3
        recommendation: "轻微划痕，标记跟踪"
      overrides: []
      overridden_by: ["scratch_major"]
)";
    auto path = WriteTempYaml(yaml);

    // 2. Build FactBase with simulated detection data
    FactBase fb;
    fb.Set("defect.type", Value::Of(std::string("scratch")),
           FactSource{FactSourceKind::Direct});
    fb.Set("defect.area_mm2", Value::Of(12.3),
           FactSource{FactSourceKind::Direct});
    fb.Set("defect.position", Value::Of(std::string("seat_center")),
           FactSource{FactSourceKind::Direct});
    fb.Set("defect.score", Value::Of(0.92),
           FactSource{FactSourceKind::Direct});
    fb.Set("defect.confidence", Value::Of(0.85),
           FactSource{FactSourceKind::Direct});

    // 3. Parse rules from YAML
    RuleEngine engine;
    auto load = engine.LoadFromYAML(path);
    ASSERT_TRUE(load.has_value()) << load.error().message;

    // 4. Evaluate all rules
    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    // 5. Resolve conflicts (overrides)
    auto resolved = engine.ResolveConflicts(*results);

    // 6. Verify scratch_major matched, scratch_minor overridden
    bool found_major = false;
    bool found_minor_not_matched = false;
    for (const auto& r : resolved) {
        if (r.name == "scratch_major") {
            found_major = r.matched;
        }
        if (r.name == "scratch_minor") {
            // scratch_minor should be present but overridden (matched=false)
            found_minor_not_matched = !r.matched;
        }
    }
    EXPECT_TRUE(found_major) << "scratch_major should match (area=12.3 > 10)";
    EXPECT_TRUE(found_minor_not_matched)
        << "scratch_minor should be overridden by scratch_major";

    std::filesystem::remove(path);
}

// ===========================================================================
// ErrorRecoverySkipsBadRule
//
// A rule whose condition references a nonexistent field should be skipped
// gracefully by the engine without crashing.
// ===========================================================================
TEST(RuleIntegrationTest, ErrorRecoverySkipsBadRule) {
    // 1. Write YAML: one good rule + one rule with missing field reference
    auto yaml = R"(
rule_sets:
  mixed_set:
    - name: "good_rule"
      priority: 50
      condition: "x > 0"
      action:
        label: "OK"
        base_severity: 0.1
        recommendation: ""
    - name: "bad_rule"
      priority: 100
      condition: "defect.nonexistent > 5"
      action:
        label: "ERROR"
        base_severity: 0.9
        recommendation: ""
)";
    auto path = WriteTempYaml(yaml);

    // 2. Setup FactBase: x=5 is set, but defect.nonexistent is NOT set
    FactBase fb;
    fb.Set("x", Value::Of(5.0),
           FactSource{FactSourceKind::Direct});

    // 3. Load YAML
    RuleEngine engine;
    auto load = engine.LoadFromYAML(path);
    ASSERT_TRUE(load.has_value()) << load.error().message;

    // 4. Evaluate all rules — should not crash
    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    // 5. All 2 rules produce a ResolvedRule entry
    EXPECT_EQ(results->size(), 2u);

    // 6. Verify good_rule matched, bad_rule skipped (error recovery)
    for (const auto& r : *results) {
        if (r.name == "good_rule") {
            EXPECT_TRUE(r.matched) << "good_rule should have matched (x=5 > 0)";
        } else if (r.name == "bad_rule") {
            EXPECT_FALSE(r.matched)
                << "bad_rule should have been skipped due to missing field";
        }
    }

    // 7. ResolveConflicts should not crash despite bad rule
    auto resolved = engine.ResolveConflicts(*results);
    ASSERT_EQ(resolved.size(), 2u);

    std::filesystem::remove(path);
}

// ===========================================================================
// ErrorRecoverySkipsBadRule_GraphPath
//
// A rule with a graph-path-style condition referencing a nonexistent path
// should also be skipped gracefully.
// ===========================================================================
TEST(RuleIntegrationTest, ErrorRecoverySkipsBadRule_GraphPath) {
    auto yaml = R"(
rule_sets:
  path_test:
    - name: "good_path_rule"
      priority: 50
      condition: "x > 0"
      action:
        label: "OK"
        base_severity: 0.1
        recommendation: ""
    - name: "bad_path_rule"
      priority: 100
      condition: "material->supplier->nonexistent > 0.5"
      action:
        label: "ERROR"
        base_severity: 0.9
        recommendation: ""
)";
    auto path = WriteTempYaml(yaml);

    FactBase fb;
    fb.Set("x", Value::Of(5.0),
           FactSource{FactSourceKind::Direct});

    RuleEngine engine;
    ASSERT_TRUE(engine.LoadFromYAML(path));

    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    EXPECT_EQ(results->size(), 2u);

    for (const auto& r : *results) {
        if (r.name == "good_path_rule") {
            EXPECT_TRUE(r.matched);
        } else if (r.name == "bad_path_rule") {
            EXPECT_FALSE(r.matched)
                << "bad_path_rule should be skipped (path not found)";
        }
    }

    std::filesystem::remove(path);
}

}  // namespace
}  // namespace sai::rule
