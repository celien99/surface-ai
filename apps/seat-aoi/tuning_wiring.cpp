#include "tuning_wiring.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/tuning/tuning_space.h>
#include <sai/tuning/tuning_objective.h>
#include <sai/tuning/tuning_scheduler.h>
#include <sai/tuning/bayesian_optimizer.h>
#include <sai/reasoner/reasoner.h>
#include <sai/pipeline/pipeline.h>
#include <sai/pipeline/pipeline_config.h>

#include "app_config.h"

// Collect all leaf nodes from the decision tree YAML in pre-order.
// A leaf node is any YAML mapping with "type: leaf".
auto CollectLeafNodes(const YAML::Node& node) -> std::vector<YAML::Node> {
    std::vector<YAML::Node> leaves;
    if (!node.IsMap()) return leaves;

    if (auto type = node["type"];
        type.IsDefined() && type.as<std::string>() == "leaf") {
        leaves.push_back(node);
        return leaves;
    }

    if (auto branches = node["branches"];
        branches.IsDefined() && branches.IsMap()) {
        for (auto it = branches.begin(); it != branches.end(); ++it) {
            auto child = CollectLeafNodes(it->second);
            leaves.insert(leaves.end(),
                          std::make_move_iterator(child.begin()),
                          std::make_move_iterator(child.end()));
        }
    }

    if (auto def = node["default"]; def.IsDefined()) {
        auto child = CollectLeafNodes(def);
        leaves.insert(leaves.end(),
                      std::make_move_iterator(child.begin()),
                      std::make_move_iterator(child.end()));
    }

    return leaves;
}

// Apply a single parameter value to the YAML tree.
// param_name formats:
//   "verdict_mapping.X"        -> root["verdict_mapping"][X] = value
//   "leaf_N.formula_M.weight_K" -> leaf_nodes[N]["formulas"][M]["weights"][K] = value
//   "leaf_N.formula_M.threshold" -> leaf_nodes[N]["formulas"][M]["threshold"] = value
auto ApplyParamToYaml(YAML::Node& root,
                      std::vector<YAML::Node>& leaf_nodes,
                      const std::string& param_name,
                      double value) -> void {
    std::vector<std::string> segs;
    {
        std::istringstream iss(param_name);
        std::string seg;
        while (std::getline(iss, seg, '.')) segs.push_back(seg);
    }
    if (segs.empty()) return;

    if (segs[0] == "verdict_mapping") {
        if (segs.size() == 2) {
            root["verdict_mapping"][segs[1]] = value;
        }
    } else if (segs.size() >= 3 && segs[0].rfind("leaf_", 0) == 0) {
        try {
            size_t li = std::stoul(segs[0].substr(5));
            if (li >= leaf_nodes.size()) return;
            auto& leaf = leaf_nodes[li];

            if (segs[1].rfind("formula_", 0) == 0) {
                size_t fi = std::stoul(segs[1].substr(8));
                if (segs[2] == "threshold") {
                    leaf["formulas"][fi]["threshold"] = value;
                } else if (segs.size() == 4 && segs[2].rfind("weight_", 0) == 0) {
                    size_t wi = std::stoul(segs[2].substr(7));
                    leaf["formulas"][fi]["weights"][wi] = value;
                }
            }
        } catch (...) {
            // Invalid segment — silently ignore
        }
    }
}

auto TryCreateTuningScheduler(
    const YAML::Node& pipeline_yaml,
    sai::knowledge::KnowledgeGraph& kg,
    sai::knowledge::KnowledgeEvolution& kg_evolution,
    sai::reasoner::IReasoner& reasoner,
    sai::pipeline::Pipeline& pipeline
) -> sai::Result<std::unique_ptr<sai::tuning::TuningScheduler>> {
    using namespace sai;
    using namespace seat_aoi::config;

    auto tuning_node = pipeline_yaml["pipeline"]["tuning"];
    if (!tuning_node.IsDefined() || !tuning_node["enabled"].as<bool>(false)) {
        return std::unique_ptr<sai::tuning::TuningScheduler>{};
    }

    auto tuning_cfg_path = TuningYaml();

    // Load tuning YAML for all config sections
    auto tuning_yaml = YAML::LoadFile(tuning_cfg_path.string());
    auto ty = tuning_yaml["tuning"];

    // 1. Parse TuningSpace
    auto space_result = tuning::TuningSpace::LoadFromYaml(tuning_cfg_path);
    if (!space_result.has_value()) {
        std::cerr << "Tuning: failed to load tuning space: "
                  << space_result.error().message << "\n";
        return std::unique_ptr<sai::tuning::TuningScheduler>{};
    }
    auto space = std::move(*space_result);
    auto space_dim = space.Dimension();

    // Capture parameter names before space is moved into BayesianOptimizer
    std::vector<std::string> param_names;
    for (const auto& p : space.Parameters()) {
        param_names.push_back(p.name);
    }

    // 2. Parse SchedulerConfig from tuning YAML
    tuning::SchedulerConfig sched_cfg;
    if (auto sn = ty["scheduler"]; sn.IsDefined()) {
        sched_cfg.interval = std::chrono::seconds(
            sn["interval_sec"].as<int>(3600));
        sched_cfg.monitoring_window = std::chrono::seconds(
            sn["monitoring_window_sec"].as<int>(300));
        sched_cfg.feedback_lookback = std::chrono::seconds(
            sn["feedback_lookback_sec"].as<int>(86400));
    }
    if (auto sn = ty["safety"]; sn.IsDefined()) {
        sched_cfg.min_ng_rate = sn["min_ng_rate"].as<double>(0.001);
        sched_cfg.max_ng_rate = sn["max_ng_rate"].as<double>(0.50);
        sched_cfg.min_samples_for_trigger =
            sn["min_samples_for_trigger"].as<std::size_t>(50);
    }

    // 3. Parse OptimizerConfig from tuning YAML
    tuning::OptimizerConfig opt_cfg;
    if (auto on = ty["optimizer"]; on.IsDefined()) {
        opt_cfg.max_iterations =
            on["max_iterations"].as<std::size_t>(50);
        opt_cfg.initial_random_points =
            on["initial_random_points"].as<std::size_t>(5);
        opt_cfg.noise_level = on["noise_level"].as<double>(0.01);
    }

    // 4. Create components
    double fp_cost = ty["objective"]["fp_cost"].as<double>(1.0);
    double fn_cost = ty["objective"]["fn_cost"].as<double>(5.0);
    auto objective = std::make_unique<tuning::KnowledgeGraphObjective>(
        kg, fp_cost, fn_cost);

    // Map verdict_mapping threshold parameter indices for simulation-aware
    // evaluation. When set, Evaluate() replays detection_score against candidate
    // thresholds instead of relying on stored machine_verdict.
    int ng_idx = -1, warn_idx = -1;
    for (std::size_t i = 0; i < param_names.size(); ++i) {
        if (param_names[i] == "verdict_mapping.ng_threshold")
            ng_idx = static_cast<int>(i);
        if (param_names[i] == "verdict_mapping.warn_threshold")
            warn_idx = static_cast<int>(i);
    }
    if (ng_idx >= 0) {
        objective->SetThresholdParamIndices(ng_idx, warn_idx);
    }

    auto optimizer = std::make_unique<tuning::BayesianOptimizer>(
        std::move(space), opt_cfg);

    auto scheduler = std::make_unique<tuning::TuningScheduler>(
        sched_cfg,
        std::move(optimizer),
        std::move(objective),
        std::shared_ptr<knowledge::KnowledgeGraph>(&kg, [](auto*) {}),
        std::shared_ptr<knowledge::KnowledgeEvolution>(&kg_evolution, [](auto*) {}));

    // 5. Inject callbacks
    auto tree_path = DecisionTree();
    scheduler->SetParameterApplier(
        [&reasoner, tree_path, param_names = std::move(param_names)]
        (const std::vector<double>& params) -> Result<void> {
            // 1. Read current YAML
            YAML::Node root;
            try {
                root = YAML::LoadFile(tree_path.string());
            } catch (const YAML::Exception& e) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Pipeline_InvalidConfig,
                    "Tuning: failed to load decision tree YAML for parameter update: " +
                        std::string(e.what())});
            }

            // 2. Collect leaf nodes in pre-order
            auto leaf_nodes = CollectLeafNodes(root);

            // 3. Apply each parameter value to its YAML path
            for (size_t i = 0; i < params.size() && i < param_names.size(); ++i) {
                ApplyParamToYaml(root, leaf_nodes, param_names[i], params[i]);
            }

            // 4. Write to temp file, then atomically rename
            auto tmp = tree_path.string() + ".tmp";
            {
                std::ofstream out(tmp);
                out << root;
            }
            std::filesystem::rename(tmp, tree_path);

            // 5. Reload tree with updated parameters
            return reasoner.ReloadTree(tree_path);
        });

    scheduler->SetMetricsPoller(
        [&pipeline]() -> Result<tuning::MetricsSnapshot> {
            tuning::MetricsSnapshot snapshot;
            auto metrics = pipeline.Metrics();
            for (const auto& sm : metrics) {
                // Use Export stage verdict counts to compute actual NG rate.
                if (sm.type == pipeline::StageType::Export) {
                    auto processed = sm.frames_processed.load();
                    auto ng = sm.frames_ng.load();
                    snapshot.sample_count += processed;
                    if (processed > 0) {
                        snapshot.ng_rate += static_cast<double>(ng);
                    }
                }
            }
            if (snapshot.sample_count > 0) {
                snapshot.ng_rate /= snapshot.sample_count;
            }
            return snapshot;
        });

    std::cout << "TuningScheduler: started with " << space_dim
              << " parameters\n";

    return std::unique_ptr<sai::tuning::TuningScheduler>(std::move(scheduler));
}
