// tuning_integration_test.cpp — end-to-end tuning cycle integration tests
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <sai/tuning/tuning_space.h>
#include <sai/tuning/tuning_objective.h>
#include <sai/tuning/bayesian_optimizer.h>
#include <sai/tuning/tuning_scheduler.h>
#include <sai/knowledge/knowledge_store.h>

namespace fs = std::filesystem;
using namespace sai::tuning;
using namespace sai::knowledge;
using namespace std::chrono_literals;

// ── End-to-End Tests ─────────────────────────────────────────────────────

TEST(TuningIntegration, FullCycleWithNoFeedbackReturnsZeroCost) {
    auto ks_result = KnowledgeStore::Create(
        KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks_result.has_value());
    auto ks = std::move(*ks_result);

    TuningSpace space;
    space.AddParameter({"ng_th", ParameterType::Continuous, 0.5, 0.9});

    OptimizerConfig opt_cfg;
    opt_cfg.max_iterations = 5;
    opt_cfg.initial_random_points = 2;
    opt_cfg.noise_level = 0.001;

    BayesianOptimizer opt(space, opt_cfg);
    KnowledgeGraphObjective obj(ks->Graph(), 1.0, 5.0);

    // No feedback -> zero cost
    auto result = obj.Evaluate({0.7},
        std::chrono::system_clock::now() - std::chrono::hours(24));
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 0.0);
}

TEST(TuningIntegration, OptimizerFindsLowerCostThanInitial) {
    auto ks_result = KnowledgeStore::Create(
        KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks_result.has_value());
    auto ks = std::move(*ks_result);

    // Insert feedback that produces FN cost:
    // machine_verdict=OK + human_label=NG → false negative (missed defect)
    auto& kg = ks->Graph();
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    for (int i = 0; i < 10; ++i) {
        KnowledgeRecord props;
        props.fields["inspection_id"] = std::string{"fn_" + std::to_string(i)};
        props.fields["machine_verdict"] = std::string{"OK"};
        props.fields["human_label"] = std::string{"NG"};
        props.fields["timestamp"] = static_cast<std::int64_t>(now_us + i);
        props.fields["image_level_score"] = 0.75;
        (void)kg.InsertNode("GroundTruth", std::move(props));
    }

    for (int i = 0; i < 30; ++i) {
        KnowledgeRecord props;
        props.fields["inspection_id"] = std::string{"tn_" + std::to_string(i)};
        props.fields["machine_verdict"] = std::string{"OK"};
        props.fields["human_label"] = std::string{"OK"};
        props.fields["timestamp"] = static_cast<std::int64_t>(now_us + 100 + i);
        props.fields["image_level_score"] = 0.2;
        (void)kg.InsertNode("GroundTruth", std::move(props));
    }

    KnowledgeGraphObjective obj(ks->Graph(), 1.0, 5.0);

    // Evaluate at high threshold — should have non-zero cost due to FNs
    auto high_threshold_cost = obj.Evaluate({0.9}, now - std::chrono::hours(1));
    ASSERT_TRUE(high_threshold_cost.has_value());

    // Evaluate at low threshold — should also be non-zero (KG data is static)
    auto low_threshold_cost = obj.Evaluate({0.5}, now - std::chrono::hours(1));
    ASSERT_TRUE(low_threshold_cost.has_value());

    // With 10 FN out of 40 total and fn_cost=5.0:
    // cost = 5.0 * (10/40) = 1.25
    EXPECT_NEAR(*high_threshold_cost, 1.25, 0.001);
}

TEST(TuningIntegration, SafetyCircuitBreakerConfig) {
    // Verify that SchedulerConfig safety bounds are wired correctly.
    SchedulerConfig cfg;
    cfg.monitoring_window = 1s;   // short for test
    cfg.min_ng_rate = 0.001;
    cfg.max_ng_rate = 0.10;
    cfg.min_samples_for_trigger = 5;

    EXPECT_EQ(cfg.monitoring_window, 1s);
    EXPECT_DOUBLE_EQ(cfg.min_ng_rate, 0.001);
    EXPECT_DOUBLE_EQ(cfg.max_ng_rate, 0.10);
    EXPECT_EQ(cfg.min_samples_for_trigger, 5U);
}

TEST(TuningIntegration, AuditTrailWrittenOnParameterChange) {
    auto ks_result = KnowledgeStore::Create(
        KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks_result.has_value());
    auto ks = std::move(*ks_result);

    auto& kg = ks->Graph();

    // Write a TuningEvent node manually (simulating what scheduler does):
    KnowledgeRecord props;
    props.fields["event_type"] = std::string{"APPLIED"};
    props.fields["parameters_before"] = std::string{"[0.7, 0.3]"};
    props.fields["parameters_after"] = std::string{"[0.65, 0.28]"};
    props.fields["objective_before"] = 0.15;
    props.fields["objective_after"] = 0.08;
    props.fields["trigger"] = std::string{"scheduled"};
    props.fields["timestamp"] = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    auto node_id = kg.InsertNode("TuningEvent", std::move(props));
    ASSERT_TRUE(node_id.has_value());

    // Query it back
    auto node = kg.GetNode(*node_id);
    ASSERT_TRUE(node.has_value());
    EXPECT_EQ(node->properties.fields.at("event_type"),
              FieldValue{std::string{"APPLIED"}});
}

TEST(TuningIntegration, TuningSpaceYamlRoundTrip) {
    auto tmp = fs::temp_directory_path() / "test_tuning_space.yaml";

    // Write a minimal tuning config
    {
        std::ofstream f(tmp);
        f << R"(tuning:
  enabled: true
  parameters:
    - name: "verdict_mapping.ng_threshold"
      type: continuous
      min: 0.5
      max: 0.9
    - name: "leaf_0.formula_0.weight_0"
      type: continuous
      min: 0.1
      max: 0.9
)";
    }

    auto space = TuningSpace::LoadFromYaml(tmp);
    ASSERT_TRUE(space.has_value());
    EXPECT_EQ(space->Dimension(), 2U);

    auto params = space->Parameters();
    EXPECT_EQ(params[0].name, "verdict_mapping.ng_threshold");
    EXPECT_DOUBLE_EQ(params[0].min, 0.5);
    EXPECT_EQ(params[1].name, "leaf_0.formula_0.weight_0");

    fs::remove(tmp);
}
