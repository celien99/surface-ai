#include <sai/scheduler/scheduler.h>

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>

namespace sai::scheduler {

using sai::pipeline::BackpressureConfig;
using sai::pipeline::StageConfig;
using sai::pipeline::StageType;

namespace {

TypeId StageTypeToPoolId(StageType t) {
    return sai::detail::Fnv1aHash(std::string(Scheduler::StageTypeToPoolKey(t)));
}

}  // anonymous namespace

// -- StageType -> pool key mapping (single source of truth) -------------

auto Scheduler::StageTypeToPoolKey(StageType t) -> std::string_view {
    switch (t) {
        case StageType::Capture:     return "Capture";
        case StageType::Preprocess:  return "Capture";
        case StageType::Inference:   return "Inference";
        case StageType::Detect:      return "Inference";
        case StageType::RuleEval:    return "Reason";
        case StageType::Reason:      return "Reason";
        case StageType::Export:      return "IO";
        case StageType::Custom:      return "Background";
    }
    return "Background";
}

auto Scheduler::PoolConfigForKey(std::string_view key) -> PoolConfig {
    if (key == "Capture")    return {2, 8};
    if (key == "Inference")  return {1, 4};
    if (key == "Reason")     return {2, 16};
    if (key == "IO")         return {1, 32};
    if (key == "Background") return {1, 16};
    return {1, 8};
}

auto Scheduler::OverridePoolConfig(std::string_view key, std::size_t threads,
                                    std::optional<std::size_t> queue_capacity) -> void {
    auto cfg = PoolConfigForKey(key);  // start from defaults
    cfg.threads = threads;
    if (queue_capacity.has_value()) {
        cfg.queue_capacity = *queue_capacity;
    }
    pool_overrides_[std::string(key)] = cfg;
}

auto Scheduler::GetEffectivePoolConfig(std::string_view key) const -> PoolConfig {
    auto it = pool_overrides_.find(key);
    if (it != pool_overrides_.end()) return it->second;
    return PoolConfigForKey(key);
}

// -- Scheduler ----------------------------------------------------------

Scheduler::Scheduler()
    : pools_(std::make_unique<Registry<runtime::WorkerPool>>()) {}

auto Scheduler::Allocate(const std::vector<StageConfig>& stages,
                          const BackpressureConfig& bp_config)
    -> Result<void> {
    bp_config_ = bp_config;
    stage_pool_map_.fill(std::nullopt);

    // Collect unique pool keys needed
    std::set<std::string> required_keys;
    for (auto& s : stages) {
        required_keys.emplace(std::string(StageTypeToPoolKey(s.type)));
    }

    // Create a WorkerPool for each unique key and register it
    for (auto& key : required_keys) {
        // Check for YAML override first, fall back to hardcoded defaults
        PoolConfig cfg;
        auto ov = pool_overrides_.find(key);
        if (ov != pool_overrides_.end()) {
            cfg = ov->second;
        } else {
            cfg = PoolConfigForKey(key);
        }
        auto pool = std::make_shared<runtime::WorkerPool>(
            cfg.threads, cfg.queue_capacity);
        TypeId type_id = sai::detail::Fnv1aHash(key);
        auto reg_result = pools_->Register(type_id, std::move(pool));
        if (!reg_result.has_value()) {
            return tl::make_unexpected(std::move(reg_result.error()));
        }
    }

    // Build stage_type -> pool_id array for O(1) PoolFor lookups
    for (auto& s : stages) {
        auto idx = static_cast<std::size_t>(s.type);
        if (idx < kStageTypeCount) {
            stage_pool_map_[idx] = StageTypeToPoolId(s.type);
        }
    }

    return {};
}

auto Scheduler::Deallocate() -> void {
    stage_pool_map_.fill(std::nullopt);
}

auto Scheduler::PoolFor(StageType type) const
    -> Result<runtime::WorkerPool*> {
    auto idx = static_cast<std::size_t>(type);
    if (idx >= kStageTypeCount || !stage_pool_map_[idx].has_value()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Scheduler_PoolNotFound,
            "StageType not allocated -- call Allocate() first"});
    }
    auto resolved = pools_->Resolve(*stage_pool_map_[idx]);
    if (!resolved.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Scheduler_PoolNotFound,
            "WorkerPool not found in registry"});
    }
    return resolved->get();
}

}  // namespace sai::scheduler
