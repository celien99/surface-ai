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
