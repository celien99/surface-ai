#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/type_id.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai {
template <typename T>
class Registry;
}  // namespace sai

namespace sai::runtime {
class WorkerPool;
}  // namespace sai::runtime

namespace sai::scheduler {

using sai::pipeline::BackpressureConfig;
using sai::pipeline::StageConfig;
using sai::pipeline::StageType;

// Scheduler manages WorkerPool allocation per StageType.
//
// Design decisions:
// - Owns the Registry<WorkerPool> that Pipeline passes to runtime::TaskScheduler.
// - Allocate(stages, bp_config) creates WorkerPools according to a fixed
//   StageType->pool mapping.
// - PoolFor(type) returns the pool for a given StageType, or an error if
//   not allocated.
class Scheduler {
public:
    Scheduler();

    // Create WorkerPools based on stage configs and register them.
    // Must be called before PoolFor().
    auto Allocate(const std::vector<StageConfig>& stages,
                  const BackpressureConfig& bp_config) -> Result<void>;

    // Clear stage->pool mappings (pools themselves remain in registry).
    auto Deallocate() -> void;

    // Return the WorkerPool for the given stage type.
    auto PoolFor(StageType type) const -> Result<runtime::WorkerPool*>;

    // Expose the underlying registry for runtime::TaskScheduler.
    auto Pools() -> Registry<runtime::WorkerPool>& { return *pools_; }

    // Map StageType to its pool string key (public for use by Pipeline).
    static auto StageTypeToPoolKey(StageType t) -> std::string_view;

    // Default pool configuration for a given pool key.
    struct PoolConfig {
        std::size_t threads;
        std::size_t queue_capacity;
    };
    static auto PoolConfigForKey(std::string_view key) -> PoolConfig;

private:
    // Fixed-size array indexed by StageType (8 values: Capture=0..Custom=7).
    // Unallocated entries hold std::nullopt — no heap allocation for lookups.
    static constexpr std::size_t kStageTypeCount = 8;
    std::unique_ptr<Registry<runtime::WorkerPool>> pools_;
    std::array<std::optional<TypeId>, kStageTypeCount> stage_pool_map_;
    BackpressureConfig bp_config_{};
};

}  // namespace sai::scheduler
