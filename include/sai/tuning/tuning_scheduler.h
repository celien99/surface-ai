// tuning_scheduler.h — TuningScheduler background thread orchestration + circuit breaker
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include <sai/core/error.h>
#include <sai/tuning/bayesian_optimizer.h>
#include <sai/tuning/tuning_objective.h>

namespace sai::knowledge {
class KnowledgeGraph;
class KnowledgeEvolution;
}  // namespace sai::knowledge

namespace sai::tuning {

enum class TuningState : std::uint8_t {
    Idle,
    Evaluating,
    Optimizing,
    Monitoring,
    RolledBack,
};

struct SchedulerConfig {
    std::chrono::seconds interval{3600};
    std::chrono::seconds monitoring_window{300};
    std::chrono::seconds feedback_lookback{86400};
    double min_ng_rate{0.001};
    double max_ng_rate{0.50};
    std::size_t min_samples_for_trigger{50};
};

class TuningScheduler final {
public:
    using ParameterApplier = std::function<Result<void>(const std::vector<double>&)>;
    using MetricsPoller = std::function<Result<double>()>;

    TuningScheduler(SchedulerConfig config,
                    std::unique_ptr<BayesianOptimizer> optimizer,
                    std::unique_ptr<ITuningObjective> objective,
                    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg,
                    std::shared_ptr<sai::knowledge::KnowledgeEvolution> evolution);

    auto SetParameterApplier(ParameterApplier fn) -> void;
    auto SetMetricsPoller(MetricsPoller fn) -> void;

    auto Start(std::stop_token token) -> void;
    auto Join() -> void;
    auto TriggerOnce() -> void;
    auto CurrentState() const -> TuningState;

    TuningScheduler(const TuningScheduler&) = delete;
    auto operator=(const TuningScheduler&) -> TuningScheduler& = delete;
    TuningScheduler(TuningScheduler&&) = delete;
    auto operator=(TuningScheduler&&) -> TuningScheduler& = delete;

private:
    auto MainLoop(std::stop_token token) -> void;

    SchedulerConfig config_;
    std::unique_ptr<BayesianOptimizer> optimizer_;
    std::unique_ptr<ITuningObjective> objective_;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> evolution_;

    ParameterApplier apply_params_;
    MetricsPoller poll_ng_rate_;

    std::jthread worker_;
    std::atomic<TuningState> state_{TuningState::Idle};
    std::atomic<bool> trigger_flag_{false};

    std::mutex mutex_;
    std::condition_variable cv_;

    std::vector<double> current_params_;
    double current_cost_{0.0};
};

}  // namespace sai::tuning
