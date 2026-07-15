// tuning_test.cpp — Bayesian Auto-Tuning unit tests
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
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

// ── BayesianOptimizer ────────────────────────────────────────────

#include <sai/tuning/bayesian_optimizer.h>
#include <sai/tuning/tuning_objective.h>

// Mock objective for deterministic testing
class MockObjective : public ITuningObjective {
public:
    explicit MockObjective(std::function<double(const std::vector<double>&)> fn)
        : fn_(std::move(fn)) {}

    auto Evaluate(const std::vector<double>& point,
                  std::chrono::system_clock::time_point) -> sai::Result<double> override {
        last_point_ = point;
        eval_count_++;
        return fn_(point);
    }

    auto EvalCount() const -> int { return eval_count_; }
    auto LastPoint() const -> const std::vector<double>& { return last_point_; }

private:
    std::function<double(const std::vector<double>&)> fn_;
    int eval_count_ = 0;
    std::vector<double> last_point_;
};

TEST(BayesianOptimizerTest, InitialRandomPointsGenerated) {
    TuningSpace space;
    space.AddParameter({"x", ParameterType::Continuous, 0.0, 1.0});

    OptimizerConfig cfg;
    cfg.max_iterations = 5;
    cfg.initial_random_points = 3;

    BayesianOptimizer opt(std::move(space), cfg);
    EXPECT_TRUE(opt.AllObservations().empty());
}

TEST(BayesianOptimizerTest, FindsMinimumOfQuadratic) {
    TuningSpace space;
    space.AddParameter({"x", ParameterType::Continuous, -5.0, 5.0});

    // Quadratic: f(x) = (x-2)^2, minimum at x=2, f(2)=0
    MockObjective obj([](const std::vector<double>& p) {
        double x = p[0];
        return (x - 2.0) * (x - 2.0);
    });

    OptimizerConfig cfg;
    cfg.max_iterations = 20;
    cfg.initial_random_points = 5;
    cfg.noise_level = 0.001;

    BayesianOptimizer opt(space, cfg);

    // Pre-seed near the optimum to speed up convergence
    opt.AddObservation({{0.0}, 4.0, std::chrono::system_clock::now()});
    opt.AddObservation({{1.0}, 1.0, std::chrono::system_clock::now()});
    opt.AddObservation({{3.0}, 1.0, std::chrono::system_clock::now()});
    opt.AddObservation({{4.0}, 4.0, std::chrono::system_clock::now()});
    opt.AddObservation({{2.0}, 0.0, std::chrono::system_clock::now()});

    auto result = opt.Optimize(obj, std::chrono::system_clock::now());
    ASSERT_TRUE(result.has_value());

    // Should converge near x=2
    EXPECT_NEAR(result->params[0], 2.0, 0.5);
    EXPECT_NEAR(result->cost, 0.0, 0.1);
}

TEST(BayesianOptimizerTest, RespectsBoundaryConstraints) {
    TuningSpace space;
    space.AddParameter({"x", ParameterType::Continuous, 0.0, 1.0});

    MockObjective obj([](const std::vector<double>& p) {
        return -p[0];  // minimize negative -> wants x as large as possible
    });

    OptimizerConfig cfg;
    cfg.max_iterations = 10;
    cfg.initial_random_points = 3;
    cfg.noise_level = 0.001;

    BayesianOptimizer opt(space, cfg);
    opt.AddObservation({{0.5}, -0.5, std::chrono::system_clock::now()});

    auto result = opt.Optimize(obj, std::chrono::system_clock::now());
    ASSERT_TRUE(result.has_value());

    // Should not exceed boundary
    EXPECT_LE(result->params[0], 1.0 + 1e-6);
}

TEST(BayesianOptimizerTest, TwoDimRosenbrock) {
    TuningSpace space;
    space.AddParameter({"x", ParameterType::Continuous, -3.0, 3.0});
    space.AddParameter({"y", ParameterType::Continuous, -3.0, 3.0});

    // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2, minimum at (1,1), f(1,1)=0
    MockObjective obj([](const std::vector<double>& p) {
        double x = p[0], y = p[1];
        double dx = 1.0 - x;
        double dy = y - x * x;
        return dx * dx + 100.0 * dy * dy;
    });

    OptimizerConfig cfg;
    cfg.max_iterations = 40;
    cfg.initial_random_points = 8;
    cfg.noise_level = 0.001;

    BayesianOptimizer opt(space, cfg);

    // Seed with points around the search space
    opt.AddObservation({{0.0, 0.0}, 1.0, std::chrono::system_clock::now()});
    opt.AddObservation({{1.0, 1.0}, 0.0, std::chrono::system_clock::now()});
    opt.AddObservation({{-1.0, 1.0}, 4.0, std::chrono::system_clock::now()});
    opt.AddObservation({{2.0, 3.0}, 1.0 + 100.0, std::chrono::system_clock::now()});
    opt.AddObservation({{0.5, 0.25}, 0.25, std::chrono::system_clock::now()});
    opt.AddObservation({{1.5, 2.0}, 0.25 + 100.0 * 0.25, std::chrono::system_clock::now()});
    opt.AddObservation({{-2.0, 4.0}, 9.0 + 100.0 * 0.0, std::chrono::system_clock::now()});
    opt.AddObservation({{3.0, 3.0}, 4.0 + 100.0 * 36.0, std::chrono::system_clock::now()});

    auto result = opt.Optimize(obj, std::chrono::system_clock::now());
    ASSERT_TRUE(result.has_value());

    // Should approach (1,1)
    EXPECT_NEAR(result->params[0], 1.0, 0.3);
    EXPECT_NEAR(result->params[1], 1.0, 0.5);
    EXPECT_NEAR(result->cost, 0.0, 0.5);
}

TEST(BayesianOptimizerTest, ConvergenceReducesCost) {
    TuningSpace space;
    space.AddParameter({"x", ParameterType::Continuous, 0.0, 10.0});

    MockObjective obj([](const std::vector<double>& p) {
        return std::sin(p[0]) + 1.0;  // minimum at x=3*pi/2 ≈ 4.71, sin=-1, cost=0
    });

    OptimizerConfig cfg;
    cfg.max_iterations = 25;
    cfg.initial_random_points = 5;
    cfg.noise_level = 0.01;

    BayesianOptimizer opt(space, cfg);

    // Uniform seeding across [0,10]
    for (int i = 0; i < 5; ++i) {
        double x = 2.0 * i;
        opt.AddObservation({{x}, std::sin(x) + 1.0, std::chrono::system_clock::now()});
    }

    auto initial_best = opt.BestPoint();
    auto result = opt.Optimize(obj, std::chrono::system_clock::now());
    ASSERT_TRUE(result.has_value());

    // After optimization, cost should be at least as good
    EXPECT_LE(result->cost, initial_best.cost + 0.1);
}
