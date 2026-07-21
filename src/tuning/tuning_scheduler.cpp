// tuning_scheduler.cpp — TuningScheduler implementation
#include <sai/tuning/tuning_scheduler.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

#include <sai/infra/logger.h>
#include <sai/knowledge/knowledge_evolution.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::tuning {

namespace {

auto SerializeParams(const std::vector<double>& params) -> std::string {
    std::ostringstream oss;
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0) oss << ",";
        oss << params[i];
    }
    return oss.str();
}

// Retry rollback with exponential backoff (3 attempts: 100ms, 200ms, 400ms).
// Returns true if rollback succeeded, false if all attempts failed.
auto RetryRollback(const TuningScheduler::ParameterApplier& apply_params,
                   const std::vector<double>& old_params) -> bool {
    static constexpr int kMaxRetries = 3;
    static constexpr auto kBaseBackoff = std::chrono::milliseconds(100);

    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        auto result = apply_params(old_params);
        if (result.has_value()) return true;
        sai::infra::Logger::Get("tuning").Log(
            sai::infra::LogLevel::Error,
            "TuningScheduler rollback attempt {}/{} failed: {}",
            attempt + 1, kMaxRetries, result.error().message);
        if (attempt < kMaxRetries - 1) {
            std::this_thread::sleep_for(kBaseBackoff * (1 << attempt));
        }
    }
    return false;
}

}  // namespace

TuningScheduler::TuningScheduler(SchedulerConfig config,
                                 std::unique_ptr<BayesianOptimizer> optimizer,
                                 std::unique_ptr<ITuningObjective> objective,
                                 std::shared_ptr<sai::knowledge::KnowledgeGraph> kg,
                                 std::shared_ptr<sai::knowledge::KnowledgeEvolution> evolution)
    : config_(std::move(config))
    , optimizer_(std::move(optimizer))
    , objective_(std::move(objective))
    , kg_(std::move(kg))
    , evolution_(std::move(evolution)) {}

auto TuningScheduler::SetParameterApplier(ParameterApplier fn) -> void {
    apply_params_ = std::move(fn);
}

auto TuningScheduler::SetMetricsPoller(MetricsPoller fn) -> void {
    poll_ng_rate_ = std::move(fn);
}

auto TuningScheduler::Start(std::stop_token token) -> void {
    worker_ = std::jthread([this](std::stop_token tok) { MainLoop(std::move(tok)); }, token);
}

auto TuningScheduler::Join() -> void {
    if (worker_.joinable()) {
        worker_.request_stop();
        cv_.notify_all();
        worker_.join();
    }
}

auto TuningScheduler::TriggerOnce() -> void {
    trigger_flag_ = true;
    cv_.notify_all();
}

auto TuningScheduler::CurrentState() const -> TuningState {
    return state_.load(std::memory_order_acquire);
}

auto TuningScheduler::MainLoop(std::stop_token token) -> void {
    while (!token.stop_requested()) {
        // ── Phase 1: Idle — wait for interval or manual trigger ──
        state_.store(TuningState::Idle, std::memory_order_release);
        {
            std::unique_lock lock(mutex_);
            auto deadline = std::chrono::steady_clock::now() + config_.interval;
            while (!trigger_flag_.load(std::memory_order_acquire)) {
                if (token.stop_requested()) return;
                auto result = cv_.wait_until(lock, deadline);
                if (result == std::cv_status::timeout) break;
            }
        }
        if (token.stop_requested()) return;
        trigger_flag_.store(false, std::memory_order_release);

        auto since = std::chrono::system_clock::now() - config_.feedback_lookback;

        // ── Phase 2: Evaluating — compute current cost ──
        state_.store(TuningState::Evaluating, std::memory_order_release);
        if (current_params_.empty()) {
            current_cost_ = 0.0;
        } else {
            auto eval_result = objective_->Evaluate(current_params_, since);
            if (!eval_result.has_value()) {
                sai::infra::Logger::Get("tuning").Log(
                    sai::infra::LogLevel::Error,
                    "TuningScheduler Evaluate failed: {}", eval_result.error().message);
                continue;
            }
            current_cost_ = *eval_result;
        }

        // ── Phase 3: Optimizing — run Bayesian optimizer ──
        state_.store(TuningState::Optimizing, std::memory_order_release);
        if (!current_params_.empty()) {
            optimizer_->AddObservation({current_params_, current_cost_,
                                         std::chrono::system_clock::now()});
        }
        auto opt_result = optimizer_->Optimize(*objective_, since);
        if (!opt_result.has_value()) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Error,
                "TuningScheduler Optimize failed: {}", opt_result.error().message);
            continue;
        }

        auto best = std::move(*opt_result);

        // Check for significant improvement (≥5% cost reduction)
        if (current_cost_ <= 0.0) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "TuningScheduler current_cost_={} <= 0 — applying new params without improvement check",
                current_cost_);
        } else if (best.cost >= current_cost_ * 0.95) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Info,
                "TuningScheduler no significant improvement: current={}, best={}",
                current_cost_, best.cost);
            continue;
        }

        // ── Phase 4: Apply new parameters ──
        auto old_params = current_params_;
        auto old_cost = current_cost_;

        // Write KG audit log: PROPOSED event
        if (kg_) {
            sai::knowledge::KnowledgeRecord props;
            props.fields["event_type"] = std::string{"PROPOSED"};
            props.fields["old_params"] = SerializeParams(old_params);
            props.fields["new_params"] = SerializeParams(best.params);
            props.fields["old_cost"] = old_cost;
            props.fields["new_cost"] = best.cost;
            props.fields["timestamp"] = static_cast<std::int64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
            auto result = kg_->InsertNode("TuningEvent", std::move(props));
            if (!result.has_value()) {
                sai::infra::Logger::Get("tuning").Log(
                    sai::infra::LogLevel::Warning,
                    "TuningScheduler failed to write KG audit: {}",
                    result.error().message);
            }
        }

        // Write KnowledgeEvolution entry
        if (evolution_) {
            sai::knowledge::KnowledgeRecord before;
            before.fields["params"] = SerializeParams(old_params);
            before.fields["cost"] = old_cost;
            auto evo_result = evolution_->Append(
                "TuningParams", 0, sai::knowledge::EvolutionOp::Update,
                std::move(before), "TuningScheduler");
            if (!evo_result.has_value()) {
                sai::infra::Logger::Get("tuning").Log(
                    sai::infra::LogLevel::Warning,
                    "TuningScheduler failed to write Evolution entry: {}",
                    evo_result.error().message);
            }
        }

        // Call parameter applier
        if (apply_params_) {
            auto apply_result = apply_params_(best.params);
            if (!apply_result.has_value()) {
                sai::infra::Logger::Get("tuning").Log(
                    sai::infra::LogLevel::Error,
                    "TuningScheduler parameter apply failed: {}",
                    apply_result.error().message);
                // Rollback to old params with retry
                if (apply_params_) {
                    if (!RetryRollback(apply_params_, old_params)) {
                        sai::infra::Logger::Get("tuning").Log(
                            sai::infra::LogLevel::Critical,
                            "TuningScheduler rollback FAILED after {} attempts",
                            3);
                    }
                }
                state_.store(TuningState::RolledBack, std::memory_order_release);
                continue;
            }
        }

        current_params_ = best.params;

        // ── Phase 5: Monitoring — poll NG rate, circuit breaker ──
        state_.store(TuningState::Monitoring, std::memory_order_release);
        auto monitor_deadline =
            std::chrono::steady_clock::now() + config_.monitoring_window;
        bool should_rollback = false;

        while (std::chrono::steady_clock::now() < monitor_deadline) {
            if (token.stop_requested()) return;

            // Sleep capped to remaining time within the monitoring window
            auto now = std::chrono::steady_clock::now();
            auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
                monitor_deadline - now);
            if (remaining.count() <= 0) break;
            std::this_thread::sleep_for(std::min(std::chrono::seconds(30), remaining));

            if (poll_ng_rate_) {
                auto ng_result = poll_ng_rate_();
                if (ng_result.has_value()) {
                    auto& snapshot = *ng_result;
                    if (snapshot.sample_count >= config_.min_samples_for_trigger) {
                        if (snapshot.ng_rate < config_.min_ng_rate ||
                            snapshot.ng_rate > config_.max_ng_rate) {
                            should_rollback = true;
                            break;
                        }
                    }
                }
            }
        }

        if (should_rollback) {
            // Rollback: restore old parameters
            state_.store(TuningState::RolledBack, std::memory_order_release);
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Error,
                "TuningScheduler NG rate out of bounds — rolling back");

            if (apply_params_) {
                if (!RetryRollback(apply_params_, old_params)) {
                    sai::infra::Logger::Get("tuning").Log(
                        sai::infra::LogLevel::Critical,
                        "TuningScheduler monitoring rollback FAILED after {} attempts",
                        3);
                }
            }
            current_params_ = old_params;

            // Write KG audit: ROLLBACK event
            if (kg_) {
                sai::knowledge::KnowledgeRecord props;
                props.fields["event_type"] = std::string{"ROLLBACK"};
                props.fields["params"] = SerializeParams(old_params);
                props.fields["rejected_params"] = SerializeParams(best.params);
                props.fields["timestamp"] = static_cast<std::int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count());
                (void)kg_->InsertNode("TuningEvent", std::move(props));
            }
        } else {
            // Commit: params kept, transition to Idle
            state_.store(TuningState::Idle, std::memory_order_release);

            // Write KG audit: APPLIED event
            if (kg_) {
                sai::knowledge::KnowledgeRecord props;
                props.fields["event_type"] = std::string{"APPLIED"};
                props.fields["old_params"] = SerializeParams(old_params);
                props.fields["new_params"] = SerializeParams(current_params_);
                props.fields["old_cost"] = old_cost;
                props.fields["new_cost"] = best.cost;
                props.fields["timestamp"] = static_cast<std::int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count());
                (void)kg_->InsertNode("TuningEvent", std::move(props));
            }

            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Info,
                "TuningScheduler cycle complete — params applied, cost {}→{}",
                old_cost, best.cost);
        }
    }
}

}  // namespace sai::tuning
