// tuning_objective.h — Batch T2: Bayesian Auto-Tuning objective function (feedback cost)
#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include <sai/core/error.h>
#include <sai/knowledge/knowledge_graph.h>

namespace sai::tuning {

struct FeedbackStats {
    std::int64_t false_positives{0};
    std::int64_t false_negatives{0};
    std::int64_t true_positives{0};
    std::int64_t true_negatives{0};
    double fp_cost_weight{1.0};
    double fn_cost_weight{1.0};

    [[nodiscard]] auto total_inspections() const -> std::int64_t {
        return false_positives + false_negatives + true_positives + true_negatives;
    }
};

class ITuningObjective {
public:
    virtual ~ITuningObjective() = default;

    [[nodiscard]] virtual auto Evaluate(const std::vector<double>& point,
                                        std::chrono::system_clock::time_point since)
        -> Result<double> = 0;
};

class KnowledgeGraphObjective final : public ITuningObjective {
public:
    KnowledgeGraphObjective(sai::knowledge::KnowledgeGraph& kg,
                            double fp_cost, double fn_cost);

    // Map parameter vector indices to the threshold params used in simulation.
    // Call before Evaluate(). When set (ng_threshold_idx >= 0), Evaluate()
    // simulates verdicts using candidate thresholds; otherwise falls back to
    // stored machine_verdict (legacy mode).
    auto SetThresholdParamIndices(int ng_threshold_idx,
                                   int warn_threshold_idx = -1) -> void;

    [[nodiscard]] auto Evaluate(const std::vector<double>& point,
                                std::chrono::system_clock::time_point since)
        -> Result<double> override;

private:
    [[nodiscard]] auto EvaluateLegacy(std::chrono::system_clock::time_point since)
        -> Result<double>;
    [[nodiscard]] auto EvaluateSimulated(const std::vector<double>& point,
                                         std::chrono::system_clock::time_point since)
        -> Result<double>;

    sai::knowledge::KnowledgeGraph& kg_;
    double fp_cost_;
    double fn_cost_;
    int ng_threshold_idx_ = -1;
    int warn_threshold_idx_ = -1;
};

}  // namespace sai::tuning
