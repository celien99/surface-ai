#include "sai/reasoner/score_calculator.h"

#include <cmath>
#include <vector>

#include "sai/rule/fact_base.h"
#include "sai/reasoner/decision_tree.h"

namespace sai::reasoner {

auto ScoreCalculator::Compute(const ScoreFormula& formula,
                              const rule::FactBase& facts) -> double {
    double sum = 0.0;
    for (size_t i = 0; i < formula.features.size(); ++i) {
        auto v = facts.Get(formula.features[i]);
        if (v.has_value()) {
            auto d = v->AsDouble();
            if (d.has_value()) {
                sum += formula.weights[i] * (*d);
            }
        }
        // Missing feature → contributes 0.0 (partial scoring)
    }
    double raw = sum - formula.threshold;
    return Sigmoid(raw);
}

auto ScoreCalculator::ComputeMax(
    const std::vector<ScoreFormula>& formulas,
    const rule::FactBase& facts) -> double {
    double best = 0.0;
    for (auto& f : formulas) {
        double s = Compute(f, facts);
        if (s > best) {
            best = s;
        }
    }
    return best;
}

double ScoreCalculator::Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace sai::reasoner
