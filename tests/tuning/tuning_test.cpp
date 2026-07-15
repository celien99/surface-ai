// tuning_test.cpp — Bayesian Auto-Tuning unit tests
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <sai/tuning/tuning_space.h>

namespace fs = std::filesystem;
using namespace sai::tuning;

// ── TuningParameter ──────────────────────────────────────────

TEST(TuningParameterTest, ContinuousParamUnconstrained) {
    TuningParameter p{"leaf_0.weight_0", ParameterType::Continuous, 0.1, 0.9};
    EXPECT_EQ(p.name, "leaf_0.weight_0");
    EXPECT_EQ(p.type, ParameterType::Continuous);
    EXPECT_DOUBLE_EQ(p.min, 0.1);
    EXPECT_DOUBLE_EQ(p.max, 0.9);
}

// ── TuningSpace ──────────────────────────────────────────────

TEST(TuningSpaceTest, EmptySpaceHasZeroDim) {
    TuningSpace space;
    EXPECT_EQ(space.Dimension(), 0U);
}

TEST(TuningSpaceTest, AddParametersReturnsCorrectDim) {
    TuningSpace space;
    space.AddParameter({"w0", ParameterType::Continuous, 0.0, 1.0});
    space.AddParameter({"w1", ParameterType::Continuous, 0.0, 1.0});
    EXPECT_EQ(space.Dimension(), 2U);
}

TEST(TuningSpaceTest, IsFeasibleWithinBounds) {
    TuningSpace space;
    space.AddParameter({"w0", ParameterType::Continuous, 0.0, 1.0});
    space.AddParameter({"th", ParameterType::Continuous, 0.1, 0.9});

    EXPECT_TRUE(space.IsFeasible({0.5, 0.3}));
    EXPECT_FALSE(space.IsFeasible({1.5, 0.3}));  // w0 out of bounds
    EXPECT_FALSE(space.IsFeasible({0.5, 0.05})); // th too low
}

TEST(TuningSpaceTest, ClampToBoundsFixesViolations) {
    TuningSpace space;
    space.AddParameter({"w0", ParameterType::Continuous, 0.0, 1.0});
    space.AddParameter({"th", ParameterType::Continuous, 0.1, 0.9});

    std::vector<double> point = {1.5, 0.05};
    space.ClampToBounds(point);
    EXPECT_DOUBLE_EQ(point[0], 1.0);
    EXPECT_DOUBLE_EQ(point[1], 0.1);
}

TEST(TuningSpaceTest, DiscreteParamRoundedInClamp) {
    TuningSpace space;
    space.AddParameter({"k", ParameterType::Discrete, 1.0, 10.0, 1.0});

    std::vector<double> point = {5.7};  // should round to 6.0
    space.ClampToBounds(point);
    EXPECT_DOUBLE_EQ(point[0], 6.0);
}

// ── FeedbackStats ────────────────────────────────────────────

#include <sai/tuning/tuning_objective.h>
#include <sai/knowledge/knowledge_store.h>

TEST(FeedbackStatsTest, WeightedCostZeroWhenEmpty) {
    FeedbackStats stats;  // all zeros
    EXPECT_EQ(stats.total_inspections(), 0);
}

// ── KnowledgeGraphObjective ──────────────────────────────────

TEST(KnowledgeGraphObjectiveTest, EmptyDBReturnsZero) {
    auto ks = sai::knowledge::KnowledgeStore::Create(
        sai::knowledge::KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks.has_value());

    KnowledgeGraphObjective obj((*ks)->Graph(), 1.0, 5.0);

    auto result = obj.Evaluate({0.5, 0.3},
        std::chrono::system_clock::now() - std::chrono::hours(24));
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 0.0);  // no data → zero cost
}

TEST(KnowledgeGraphObjectiveTest, PerfectPredictionGetsZeroCost) {
    auto ks = sai::knowledge::KnowledgeStore::Create(
        sai::knowledge::KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks.has_value());

    // Insert GroundTruth: machine verdict matches human label for all
    auto& kg = (*ks)->Graph();
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    for (int i = 0; i < 10; ++i) {
        sai::knowledge::KnowledgeRecord props;
        props.fields["inspection_id"] = std::string{"insp_" + std::to_string(i)};
        props.fields["machine_verdict"] = std::string{"OK"};
        props.fields["human_label"] = std::string{"OK"};
        props.fields["timestamp"] = static_cast<std::int64_t>(now_us + i);
        (void)kg.InsertNode("GroundTruth", std::move(props));
    }

    KnowledgeGraphObjective obj((*ks)->Graph(), 1.0, 5.0);

    auto result = obj.Evaluate({0.5, 0.3}, now - std::chrono::hours(1));
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(*result, 0.0);  // all correct → zero cost
}

TEST(KnowledgeGraphObjectiveTest, HighFNRateReturnsHighCost) {
    auto ks = sai::knowledge::KnowledgeStore::Create(
        sai::knowledge::KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks.has_value());

    auto& kg = (*ks)->Graph();
    auto now = std::chrono::system_clock::now();
    auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    // Insert 8 OK→OK (correct) + 2 NG→OK (missed detections = FN)
    for (int i = 0; i < 8; ++i) {
        sai::knowledge::KnowledgeRecord props;
        props.fields["inspection_id"] = std::string{"ok_" + std::to_string(i)};
        props.fields["machine_verdict"] = std::string{"OK"};
        props.fields["human_label"] = std::string{"OK"};
        props.fields["timestamp"] = static_cast<std::int64_t>(now_us + i);
        (void)kg.InsertNode("GroundTruth", std::move(props));
    }
    for (int i = 0; i < 2; ++i) {
        sai::knowledge::KnowledgeRecord props;
        props.fields["inspection_id"] = std::string{"fn_" + std::to_string(i)};
        props.fields["machine_verdict"] = std::string{"OK"};
        props.fields["human_label"] = std::string{"NG"};
        props.fields["timestamp"] = static_cast<std::int64_t>(now_us + 100 + i);
        (void)kg.InsertNode("GroundTruth", std::move(props));
    }

    // fn_cost = 5.0, fn_rate = 2/10 = 0.2 → cost = 5.0 * 0.2 = 1.0
    KnowledgeGraphObjective obj((*ks)->Graph(), 1.0, 5.0);

    auto result = obj.Evaluate({0.5, 0.3}, now - std::chrono::hours(1));
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1.0, 0.001);  // fn_cost * fn_rate
}
