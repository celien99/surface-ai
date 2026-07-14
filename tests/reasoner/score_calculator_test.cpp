#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sai/reasoner/score_calculator.h"
#include "sai/rule/fact_base.h"

namespace sai::reasoner {
namespace {

// ===========================================================================
// Sigmoid — mathematical correctness
// ===========================================================================

TEST(ScoreCalculatorTest, Sigmoid_AtZero) {
    // sigmoid(0) = 1.0 / (1.0 + exp(0)) = 1.0 / 2.0 = 0.5 exactly
    EXPECT_DOUBLE_EQ(ScoreCalculator::Sigmoid(0.0), 0.5);
}

TEST(ScoreCalculatorTest, Sigmoid_LargePositive) {
    // sigmoid(10) ~ 0.9999546...
    auto s = ScoreCalculator::Sigmoid(10.0);
    EXPECT_GT(s, 0.9999);
    EXPECT_LT(s, 1.0);
}

TEST(ScoreCalculatorTest, Sigmoid_LargeNegative) {
    // sigmoid(-10) ~ 0.0000454...
    auto s = ScoreCalculator::Sigmoid(-10.0);
    EXPECT_GT(s, 0.0);
    EXPECT_LT(s, 0.0001);
}

// ===========================================================================
// Compute — single formula evaluation
// ===========================================================================

TEST(ScoreCalculatorTest, Compute_SingleFormula) {
    // weights=[0.6, 0.4], features=["a", "b"], threshold=0.5
    // a=1.0, b=0.5  →  sum = 0.6*1.0 + 0.4*0.5 = 0.8
    // raw = 0.8 - 0.5 = 0.3  →  sigmoid(0.3) ≈ 0.5744
    ScoreFormula formula{{0.6, 0.4}, {"a", "b"}, 0.5};

    rule::FactBase facts;
    facts.Set("a", rule::Value::Of(1.0),
              {rule::FactSourceKind::Direct, "test"});
    facts.Set("b", rule::Value::Of(0.5),
              {rule::FactSourceKind::Direct, "test"});

    double expected = 1.0 / (1.0 + std::exp(-0.3));
    double result = ScoreCalculator::Compute(formula, facts);
    EXPECT_NEAR(result, expected, 1e-9);
}

TEST(ScoreCalculatorTest, Compute_MissingFeatureContributesZero) {
    // weights=[1.0, 2.0], features=["a", "b"], threshold=0.3
    // Only "a"=0.5 is set. Missing "b" contributes 0.0.
    // sum = 1.0*0.5 + 2.0*0.0 = 0.5
    // raw = 0.5 - 0.3 = 0.2  →  sigmoid(0.2)
    ScoreFormula formula{{1.0, 2.0}, {"a", "b"}, 0.3};

    rule::FactBase facts;
    facts.Set("a", rule::Value::Of(0.5),
              {rule::FactSourceKind::Direct, "test"});
    // "b" is intentionally missing

    double expected = 1.0 / (1.0 + std::exp(-0.2));
    double result = ScoreCalculator::Compute(formula, facts);
    EXPECT_NEAR(result, expected, 1e-9);
}

// ===========================================================================
// ComputeMax — selects the highest score across formulas
// ===========================================================================

TEST(ScoreCalculatorTest, ComputeMax_ReturnsBestFormula) {
    // formula1: threshold=0.5, x=0.5 → score = sigmoid(0) = 0.5
    // formula2: threshold=0.0, y=0.8 → score = sigmoid(0.8) ≈ 0.6900
    ScoreFormula f1{{1.0}, {"x"}, 0.5};
    ScoreFormula f2{{1.0}, {"y"}, 0.0};

    rule::FactBase facts;
    facts.Set("x", rule::Value::Of(0.5),
              {rule::FactSourceKind::Direct, "test"});
    facts.Set("y", rule::Value::Of(0.8),
              {rule::FactSourceKind::Direct, "test"});

    double max = ScoreCalculator::ComputeMax({f1, f2}, facts);
    double expected = 1.0 / (1.0 + std::exp(-0.8));
    EXPECT_NEAR(max, expected, 1e-9);
}

}  // namespace
}  // namespace sai::reasoner
