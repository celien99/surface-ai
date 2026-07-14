#pragma once

#include <filesystem>
#include <vector>

#include <sai/core/error.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai::pipeline {

// Internal: translates YAML → PipelineConfig, validates, builds TaskGraph.
// Not exposed in include/sai/ — Pipeline::LoadFromYAML is the public entry point.
class PipelineBuilder {
public:
    static auto ParseFromYAML(std::filesystem::path yaml_path)
        -> Result<PipelineConfig>;

    static auto Validate(const PipelineConfig& config) -> Result<void>;

private:
    // Topological sort (Kahn's algorithm) — returns true if acyclic.
    static auto IsAcyclic(const std::vector<StageConfig>& stages) -> bool;

    // Map from StageType → expected output type index in StageOutput variant
    static auto OutputTypeIndex(StageType) -> std::size_t;
    static auto InputTypeIndex(StageType) -> std::size_t;

    // Check adjacent stages have compatible types
    static auto CheckTypeCompatibility(
        const StageConfig& upstream,
        const StageConfig& downstream) -> bool;

    static auto ParseStageConfig(YAML::Node& node,
        const BackpressureConfig& bp) -> Result<StageConfig>;

    static auto MergeBackpressure(const BackpressureConfig& bp,
        const std::string& stage_id) -> BackpressurePolicy;
};

}  // namespace sai::pipeline
