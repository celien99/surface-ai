#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "sai/rule/rule_engine.h"
#include "sai/rule/value.h"
#include "sai/core/error.h"

namespace sai::rule {
namespace {

// Helper — write a string to a temp file and return the path
auto WriteTempYaml(const std::string& content) -> std::filesystem::path {
    auto tmp = std::filesystem::temp_directory_path() /
               ("sai_rule_integration_" + std::to_string(std::rand()) + ".yaml");
    std::ofstream ofs(tmp);
    ofs << content;
    ofs.close();
    return tmp;
}

// EndToEndRuleEvaluation — Full pipeline: FactBase → EvaluateAll → ResolveConflicts
TEST(RuleIntegrationTest, EndToEndRuleEvaluation) {
    auto yaml = R"(
rule_sets:
  scratch_inspection:
    - name: "scratch_major"
      priority: 100
      condition:
        and:
          - field: defect.type
            op: eq
            value: scratch
          - field: defect.area_mm2
            op: gt
            value: 10
      action:
        label: "scratch_major"
        base_severity: 0.8
        recommendation: "Major scratch — rework required"
      overrides: ["scratch_minor"]
      overridden_by: []
    - name: "scratch_minor"
      priority: 50
      condition:
        and:
          - field: defect.type
            op: eq
            value: scratch
          - field: defect.area_mm2
            op: gt
            value: 2
      action:
        label: "scratch_minor"
        base_severity: 0.3
        recommendation: "Minor scratch — track"
      overrides: []
      overridden_by: ["scratch_major"]
)";
    auto path = WriteTempYaml(yaml);

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

    RuleEngine engine;
    auto load = engine.LoadFromYAML(path);
    ASSERT_TRUE(load.has_value()) << load.error().message;

    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    auto resolved = engine.ResolveConflicts(*results);

    bool found_major = false;
    bool found_minor_not_matched = false;
    for (const auto& r : resolved) {
        if (r.name == "scratch_major") found_major = r.matched;
        if (r.name == "scratch_minor") found_minor_not_matched = !r.matched;
    }
    EXPECT_TRUE(found_major) << "scratch_major should match (area=12.3 > 10)";
    EXPECT_TRUE(found_minor_not_matched)
        << "scratch_minor should be overridden by scratch_major";

    std::filesystem::remove(path);
}

// ErrorRecoverySkipsBadRule — rule with nonexistent field should be skipped
TEST(RuleIntegrationTest, ErrorRecoverySkipsBadRule) {
    auto yaml = R"(
rule_sets:
  mixed_set:
    - name: "good_rule"
      priority: 50
      condition:
        field: x
        op: gt
        value: 0
      action:
        label: "OK"
        base_severity: 0.1
        recommendation: ""
    - name: "bad_rule"
      priority: 100
      condition:
        field: defect.nonexistent
        op: gt
        value: 5
      action:
        label: "ERROR"
        base_severity: 0.9
        recommendation: ""
)";
    auto path = WriteTempYaml(yaml);

    FactBase fb;
    fb.Set("x", Value::Of(5.0), FactSource{FactSourceKind::Direct});

    RuleEngine engine;
    auto load = engine.LoadFromYAML(path);
    ASSERT_TRUE(load.has_value()) << load.error().message;

    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value()) << results.error().message;

    EXPECT_EQ(results->size(), 2u);

    for (const auto& r : *results) {
        if (r.name == "good_rule") {
            EXPECT_TRUE(r.matched) << "good_rule should have matched (x=5 > 0)";
        } else if (r.name == "bad_rule") {
            EXPECT_FALSE(r.matched)
                << "bad_rule should have been skipped due to missing field";
        }
    }

    auto resolved = engine.ResolveConflicts(*results);
    ASSERT_EQ(resolved.size(), 2u);

    std::filesystem::remove(path);
}

// ErrorRecoverySkipsBadRule_GraphPath — rule with nonexistent path should be skipped
TEST(RuleIntegrationTest, ErrorRecoverySkipsBadRule_GraphPath) {
    auto yaml = R"(
rule_sets:
  path_test:
    - name: "good_path_rule"
      priority: 50
      condition:
        field: x
        op: gt
        value: 0
      action:
        label: "OK"
        base_severity: 0.1
        recommendation: ""
    - name: "bad_path_rule"
      priority: 100
      condition:
        path: "material->supplier->nonexistent"
        op: gt
        value: 0.5
      action:
        label: "ERROR"
        base_severity: 0.9
        recommendation: ""
)";
    auto path = WriteTempYaml(yaml);

    FactBase fb;
    fb.Set("x", Value::Of(5.0), FactSource{FactSourceKind::Direct});

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
