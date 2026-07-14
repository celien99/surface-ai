#include "scheduler.h"

#include <set>

#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>

namespace sai::pipeline {

namespace {

TypeId StageTypeToPoolId(StageType t) {
    switch (t) {
        case StageType::Capture:     return sai::detail::Fnv1aHash("Capture");
        case StageType::Preprocess:  return sai::detail::Fnv1aHash("Capture");
        case StageType::Inference:   return sai::detail::Fnv1aHash("Inference");
        case StageType::Detect:      return sai::detail::Fnv1aHash("Inference");
        case StageType::RuleEval:    return sai::detail::Fnv1aHash("Reason");
        case StageType::Reason:      return sai::detail::Fnv1aHash("Reason");
        case StageType::Export:      return sai::detail::Fnv1aHash("IO");
        case StageType::Custom:      return sai::detail::Fnv1aHash("Background");
    }
    return sai::detail::Fnv1aHash("Background");
}

}  // anonymous namespace

Scheduler::Scheduler(Registry<runtime::WorkerPool>& pools,
                     const BackpressureConfig& bp_config)
    : pools_(pools), bp_config_(bp_config) {}

auto Scheduler::Allocate(const std::vector<StageConfig>& stages)
    -> Result<void> {
    stage_pool_map_.clear();

    // Map each stage type to its pool TypeId
    std::set<TypeId> required_pools;
    for (auto& s : stages) {
        auto pool_id = StageTypeToPoolId(s.type);
        required_pools.insert(pool_id);
        stage_pool_map_[s.type] = pool_id;
    }

    // Verify all required pools exist in the registry
    for (auto& pool_id : required_pools) {
        auto resolved = pools_.Resolve(pool_id);
        if (!resolved.has_value()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Scheduler_PoolNotFound,
                "No WorkerPool registered for stage type"});
        }
    }

    return {};
}

auto Scheduler::Deallocate() -> Result<void> {
    stage_pool_map_.clear();
    return {};
}

auto Scheduler::PoolFor(StageType type) const
    -> Result<runtime::WorkerPool*> {
    auto it = stage_pool_map_.find(type);
    if (it == stage_pool_map_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Scheduler_PoolNotFound,
            "StageType not allocated"});
    }
    // Resolve from registry (returns shared_ptr<WorkerPool>)
    auto resolved = pools_.Resolve(it->second);
    if (!resolved.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Scheduler_PoolNotFound,
            "WorkerPool not found in registry"});
    }
    return resolved->get();
}

}  // namespace sai::pipeline
