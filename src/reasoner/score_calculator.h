#pragma once

#include <vector>

#include "sai/rule/fact_base.h"
#include "sai/reasoner/decision_tree.h"  // ScoreFormula

namespace sai::reasoner {

// -----------------------------------------------------------------------
// ScoreCalculator — evaluate ScoreFormula instances against a FactBase
// -----------------------------------------------------------------------
class ScoreCalculator {
public:
    /// Compute the score for a single formula against the given facts.
    /// Missing features contribute 0.0 (partial scoring).
    static auto Compute(const ScoreFormula& formula,
                        const rule::FactBase& facts) -> double;

    /// Compute the maximum score across all formulas.
    static auto ComputeMax(const std::vector<ScoreFormula>& formulas,
                           const rule::FactBase& facts) -> double;

    /// Standard logistic sigmoid: 1.0 / (1.0 + exp(-x))
    static auto Sigmoid(double x) -> double;
};

}  // namespace sai::reasoner
