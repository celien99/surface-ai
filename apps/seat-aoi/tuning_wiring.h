#pragma once

#include <optional>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <sai/core/error.h>
#include <sai/tuning/tuning_scheduler.h>

namespace sai {
namespace knowledge { class KnowledgeGraph; class KnowledgeEvolution; }
namespace reasoner { class IReasoner; }
namespace pipeline { class Pipeline; }
}

// Collect all leaf nodes from the decision tree YAML in pre-order.
// A leaf node is any YAML mapping with "type: leaf".
auto CollectLeafNodes(const YAML::Node& node) -> std::vector<YAML::Node>;

// Apply a single parameter value to the YAML tree.
// param_name formats:
//   "verdict_mapping.X"        -> root["verdict_mapping"][X] = value
//   "leaf_N.formula_M.weight_K" -> leaf_nodes[N]["formulas"][M]["weights"][K] = value
//   "leaf_N.formula_M.threshold" -> leaf_nodes[N]["formulas"][M]["threshold"] = value
auto ApplyParamToYaml(YAML::Node& root,
                      std::vector<YAML::Node>& leaf_nodes,
                      const std::string& param_name,
                      double value) -> void;

auto TryCreateTuningScheduler(
    const YAML::Node& pipeline_yaml,
    sai::knowledge::KnowledgeGraph& kg,
    sai::knowledge::KnowledgeEvolution& kg_evolution,
    sai::reasoner::IReasoner& reasoner,
    sai::pipeline::Pipeline& pipeline
) -> sai::Result<std::unique_ptr<sai::tuning::TuningScheduler>>;
