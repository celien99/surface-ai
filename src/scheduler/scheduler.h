#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/type_id.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai {
template <typename T> class Registry;
}  // namespace sai

namespace sai::runtime {
class WorkerPool;
}  // namespace sai::runtime

namespace sai::pipeline {

class Scheduler {
public:
    explicit Scheduler(Registry<runtime::WorkerPool>& pools,
                       const BackpressureConfig& bp_config);

    auto Allocate(const std::vector<StageConfig>& stages) -> Result<void>;
    auto Deallocate() -> Result<void>;
    auto PoolFor(StageType type) const -> Result<runtime::WorkerPool*>;

private:
    Registry<runtime::WorkerPool>& pools_;
    BackpressureConfig bp_config_;
    std::map<StageType, TypeId> stage_pool_map_;
};

}  // namespace sai::pipeline
