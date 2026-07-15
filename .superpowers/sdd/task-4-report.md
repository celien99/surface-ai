# Task 4 Report: TuningScheduler -- background thread orchestration + circuit breaker

## Summary

Task 4 created the `TuningScheduler` class that orchestrates the complete Bayesian auto-tuning cycle in a background thread, with circuit breaker monitoring and automatic rollback.

## Files Created / Modified

| File | Action |
|------|--------|
| `include/sai/tuning/tuning_scheduler.h` | Created (78 lines) |
| `src/tuning/tuning_scheduler.cpp` | Created (220 lines) |
| `src/tuning/CMakeLists.txt` | Modified (added tuning_scheduler.cpp) |
| `tests/tuning/tuning_test.cpp` | Modified (appended 3 tests, +63 lines) |

## Key Interfaces Produced

### TuningState enum

```cpp
enum class TuningState : std::uint8_t {
    Idle, Evaluating, Optimizing, Monitoring, RolledBack
};
```

### SchedulerConfig struct

Default values match the spec:
- `interval`: 1 hour
- `monitoring_window`: 5 minutes
- `feedback_lookback`: 24 hours
- `min_ng_rate`: 0.001
- `max_ng_rate`: 0.50
- `min_samples_for_trigger`: 50

### TuningScheduler class

- Constructor takes `SchedulerConfig`, `unique_ptr<BayesianOptimizer>`, `unique_ptr<ITuningObjective>`, `shared_ptr<KnowledgeGraph>`, `shared_ptr<KnowledgeEvolution>`
- `SetParameterApplier(fn)` / `SetMetricsPoller(fn)` inject callbacks
- `Start(stop_token)` spins up `std::jthread`
- `Join()` requests stop, notifies CV, joins thread
- `TriggerOnce()` forces immediate tuning cycle
- `CurrentState()` returns atomic `TuningState`

## Main Loop Implementation

The worker thread executes this cycle:

1. **Idle**: Wait `config.interval` on condition_variable, or wake on `TriggerOnce()` or stop request
2. **Evaluating**: Query `objective_->Evaluate()` with current params and lookback window
3. **Optimizing**: `optimizer_->AddObservation(current)`, then `optimizer_->Optimize()` to find best params
4. **Apply**: If `best.cost < current.cost * 0.95` (5% improvement), write KG audit log (`TuningEvent` node with PROPOSED snapshot), write `KnowledgeEvolution` entry, call `apply_params_(best.params)`
5. **Monitoring**: Poll `poll_ng_rate_()` every 30s during `monitoring_window`; if NG rate outside `[min_ng_rate, max_ng_rate]`, trigger rollback
6. **Rollback**: If out-of-bounds, call `apply_params_(old_params)`, write ROLLBACK `TuningEvent`, restore old params
7. **Commit**: Else write APPLIED `TuningEvent`, state returns to Idle

## Platform Adaptation

Apple libc++ does not implement `std::condition_variable::wait_for` with `std::stop_token` (C++20). The Idle wait loop uses a manual `wait_until` loop with an explicit `token.stop_requested()` check instead. This is functionally equivalent and portable.

## Dependencies

The scheduler uses:
- `sai::knowledge::KnowledgeGraph` for audit logging (`TuningEvent` nodes)
- `sai::knowledge::KnowledgeEvolution` for change records (`EvolutionOp::Update`)
- `sai::infra::Logger` for operational logging at Info/Warning/Error levels
- `sqlite3` (forward-declared in header, full include only in .cpp)

No new CMake link dependencies were needed -- `sai::knowledge` was already linked from the tuning target.

## Test Results

**18/18 tests pass**, including 3 new tests:

1. **SchedulerConfigTest.DefaultValues** -- verifies all 6 default config values match spec
2. **TuningStateTest.DistinctValues** -- verifies all 5 enum values are distinct
3. **TuningSchedulerTest.InitialStateIsIdle** -- constructs scheduler with in-memory KnowledgeGraph/KnowledgeEvolution, verifies `CurrentState() == TuningState::Idle`

## Build Verification

```bash
cmake --build --preset default  # 100% success, no warnings
ctest --preset default -I 550,567 --output-on-failure  # 18/18 passed
```

## Post-Task Fixes (2026-07-15)

6 issues fixed after initial implementation (commit 926b2f3):

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| C1 | Critical | Destructor deadlock: implicit `~jthread()` does not notify CV, worker hangs forever on Apple libc++ | Added `~TuningScheduler() { Join(); }` to header; `Join()` calls `cv_.notify_all()` |
| C2 | Critical | `min_samples_for_trigger` (default 50) never checked — monitoring loop triggered on any poll regardless of sample count | Added `MetricsSnapshot { ng_rate, sample_count }` struct; changed `MetricsPoller` from `Result<double>()` to `Result<MetricsSnapshot>()`; monitoring loop checks `snapshot.sample_count >= config_.min_samples_for_trigger` |
| I3 | Important | When `current_cost_ <= 0.0`, improvement check (5% threshold) skipped entirely, always applying new params silently | Added Warning-level log: "current_cost_={} <= 0 -- applying new params without improvement check" |
| I4 | Important | Monitoring loop overshoots deadline — `sleep_for(30s)` unconditionally even when `<30s` remain | Sleep capped to `std::min(30s, deadline - now)` with early break when remaining <= 0 |
| I5 | Important | Failed rollback silently discards error with `(void)` cast at both rollback sites (apply-failure + monitoring) | Replaced `(void)` with Error-level log at both sites: "rollback failed: {error}" and "monitoring rollback failed: {error}" |
| M7 | Minor | Redundant null check: inner `if (apply_params_)` inside outer `if (apply_params_)` block in the apply-failure rollback path | Removed the inner redundant `if (apply_params_)` guard |
