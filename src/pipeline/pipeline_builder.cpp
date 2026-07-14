#include "pipeline_builder.h"

#include <set>
#include <map>
#include <queue>
#include <algorithm>

#include <sai/core/error.h>
#include <tl/expected.hpp>

namespace sai::pipeline {

auto PipelineBuilder::ParseFromYAML(std::filesystem::path yaml_path)
    -> Result<PipelineConfig> {
    try {
        YAML::Node root = YAML::LoadFile(yaml_path.string());
        auto pipe = root["pipeline"];
        if (!pipe.IsDefined()) {
            return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "YAML root must contain 'pipeline' key"});
        }

        PipelineConfig config;
        config.name = pipe["name"].as<std::string>();
        config.version = pipe["version"].as<std::string>();

        // Parse backpressure
        if (auto bp = pipe["backpressure"]; bp.IsDefined()) {
            auto default_str = bp["default"].as<std::string>("block");
            config.backpressure.default_policy =
                BackpressurePolicyFromString(default_str);

            if (auto overrides = bp["stage_overrides"]; overrides.IsDefined()) {
                for (auto it = overrides.begin(); it != overrides.end(); ++it) {
                    config.backpressure.stage_overrides[
                        it->first.as<std::string>()] =
                        BackpressurePolicyFromString(it->second.as<std::string>());
                }
            }
        }

        // Parse stages
        auto stages_node = pipe["stages"];
        for (auto stage_node : stages_node) {
            auto stage = ParseStageConfig(stage_node, config.backpressure);
            if (!stage.has_value()) return tl::make_unexpected(std::move(stage).error());
            config.stages.push_back(std::move(*stage));
        }

        return config;
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig, e.what()});
    } catch (const std::invalid_argument& e) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig, e.what()});
    }
}

auto PipelineBuilder::Validate(const PipelineConfig& config) -> Result<void> {
    if (config.name.empty()) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline.name is required"});
    }
    if (config.stages.empty()) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline must have at least one stage"});
    }

    // Check unique ids
    std::set<std::string> ids;
    for (auto& s : config.stages) {
        if (s.id.empty()) {
            return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "stage id must be non-empty"});
        }
        if (!ids.insert(s.id).second) {
            return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "duplicate stage id: " + s.id});
        }
    }

    // Check depends_on references exist
    for (auto& s : config.stages) {
        for (auto& dep : s.depends_on) {
            if (ids.find(dep) == ids.end()) {
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                    "stage '" + s.id + "' depends_on '" + dep
                    + "' which does not exist"});
            }
        }
    }

    // Check acyclic
    if (!IsAcyclic(config.stages)) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline has a cyclic dependency"});
    }

    // Check entry: at least one stage with empty depends_on
    bool has_entry = false;
    for (auto& s : config.stages) {
        if (s.depends_on.empty()) { has_entry = true; break; }
    }
    if (!has_entry) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "no entry stage (empty depends_on) found"});
    }

    // Check exit: at least one stage not depended on by any other
    std::set<std::string> depended_on;
    for (auto& s : config.stages) {
        for (auto& d : s.depends_on) depended_on.insert(d);
    }
    bool has_exit = false;
    for (auto& s : config.stages) {
        if (depended_on.find(s.id) == depended_on.end()) {
            has_exit = true; break;
        }
    }
    if (!has_exit) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "no exit stage found"});
    }

    // Check type compatibility for each edge
    for (auto& downstream : config.stages) {
        for (auto& dep_id : downstream.depends_on) {
            // Find upstream
            auto it = std::find_if(config.stages.begin(), config.stages.end(),
                [&](auto& s) { return s.id == dep_id; });
            if (it != config.stages.end()) {
                if (!CheckTypeCompatibility(*it, downstream)) {
                    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
                        "type mismatch: " + it->id + " → " + downstream.id});
                }
            }
        }
    }

    return {};
}

// Private helpers

auto PipelineBuilder::IsAcyclic(const std::vector<StageConfig>& stages) -> bool {
    // Kahn's algorithm
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> adj;

    for (auto& s : stages) {
        in_degree[s.id] = 0;  // ensure all nodes in map
    }
    for (auto& s : stages) {
        for (auto& dep : s.depends_on) {
            adj[dep].push_back(s.id);
            in_degree[s.id]++;
        }
    }

    std::queue<std::string> q;
    for (auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    size_t visited = 0;
    while (!q.empty()) {
        auto id = q.front(); q.pop();
        visited++;
        for (auto& next : adj[id]) {
            if (--in_degree[next] == 0) q.push(next);
        }
    }

    return visited == stages.size();
}

auto PipelineBuilder::CheckTypeCompatibility(
    const StageConfig& upstream, const StageConfig& downstream) -> bool {
    return OutputTypeIndex(upstream.type) == InputTypeIndex(downstream.type);
}

auto PipelineBuilder::OutputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage
        case StageType::Preprocess:  return 1;  // SurfaceImage
        case StageType::Inference:   return 2;  // DetectionResult
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 3;  // ResolvedRule[]
        case StageType::Reason:      return 4;  // ReasoningResult
        case StageType::Export:      return 4;  // ReasoningResult (in)
        case StageType::Custom:      return 0;  // flexible, don't check here
    }
    return 0;
}

auto PipelineBuilder::InputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage (external Submit)
        case StageType::Preprocess:  return 0;  // RawImage
        case StageType::Inference:   return 1;  // SurfaceImage
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 2;  // DetectionResult
        case StageType::Reason:      return 3;  // ResolvedRule[]
        case StageType::Export:      return 4;  // ReasoningResult
        case StageType::Custom:      return 0;  // flexible
    }
    return 0;
}

auto PipelineBuilder::ParseStageConfig(const YAML::Node& node,
    const BackpressureConfig& bp) -> Result<StageConfig> {
    StageConfig stage;
    stage.id = node["id"].as<std::string>();
    stage.type = StageTypeFromString(node["type"].as<std::string>());

    if (auto deps = node["depends_on"]; deps.IsDefined()) {
        for (auto d : deps) {
            stage.depends_on.push_back(d.as<std::string>());
        }
    }

    stage.config = node["config"];  // pass through

    // Merge backpressure: per-stage override > default
    stage.backpressure = MergeBackpressure(bp, stage.id);

    if (auto qc = node["queue_capacity"]; qc.IsDefined()) {
        stage.queue_capacity = qc.as<std::size_t>();
    }

    return stage;
}

auto PipelineBuilder::MergeBackpressure(const BackpressureConfig& bp,
    const std::string& stage_id) -> BackpressurePolicy {
    auto it = bp.stage_overrides.find(stage_id);
    if (it != bp.stage_overrides.end()) return it->second;
    return bp.default_policy;
}

}  // namespace sai::pipeline
