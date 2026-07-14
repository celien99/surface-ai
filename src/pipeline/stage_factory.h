#pragma once

#include <memory>
#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace sai::pipeline {

class StageFactory {
public:
    static auto Create(const StageConfig& config, Context& ctx)
        -> Result<std::unique_ptr<IStageNode>>;
};

}  // namespace sai::pipeline
