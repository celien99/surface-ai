#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::pipeline {

// Multi-position detector key: (surface_id, position_id).
// Owned by pipeline_config.h as a pipeline-level routing concept —
// shared by stage_nodes.h (DetectStage) and app_builder.h (AssembledApp).
using BankKey = std::pair<std::string, std::uint16_t>;

enum class StageType {
    Capture,
    Preprocess,
    Inference,
    Detect,
    RuleEval,
    Reason,
    Export,
    Custom
};

inline auto StageTypeFromString(std::string_view s) -> StageType {
    if (s == "Capture")     return StageType::Capture;
    if (s == "Preprocess")  return StageType::Preprocess;
    if (s == "Inference")   return StageType::Inference;
    if (s == "Detect")      return StageType::Detect;
    if (s == "RuleEval")    return StageType::RuleEval;
    if (s == "Reason")      return StageType::Reason;
    if (s == "Export")      return StageType::Export;
    if (s == "Custom")      return StageType::Custom;
    throw std::invalid_argument(std::string("Unknown StageType: ") + std::string(s));
}

enum class BackpressurePolicy {
    Block,
    DropOldest,
    Degrade  // v1 预留，实现仅 Block/DropOldest
};

inline auto BackpressurePolicyFromString(std::string_view s) -> BackpressurePolicy {
    if (s == "block")        return BackpressurePolicy::Block;
    if (s == "drop_oldest")  return BackpressurePolicy::DropOldest;
    if (s == "degrade")      return BackpressurePolicy::Degrade;
    throw std::invalid_argument(std::string("Unknown BackpressurePolicy: ") + std::string(s));
}

struct StageConfig {
    std::string id;
    StageType type;
    std::vector<std::string> depends_on;
    YAML::Node config;
    BackpressurePolicy backpressure = BackpressurePolicy::Block;  // 加载时从 PipelineConfig 合并
    std::optional<std::size_t> queue_capacity;
};

struct BackpressureConfig {
    BackpressurePolicy default_policy = BackpressurePolicy::Block;
    std::map<std::string, BackpressurePolicy> stage_overrides;
};

struct PipelineConfig {
    std::string name;
    std::string version;
    BackpressureConfig backpressure;
    std::vector<StageConfig> stages;
};

}  // namespace sai::pipeline
