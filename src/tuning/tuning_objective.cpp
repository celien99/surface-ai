#include <sai/tuning/tuning_objective.h>

#include <cstdint>
#include <string>

#include <sai/infra/logger.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::tuning {

KnowledgeGraphObjective::KnowledgeGraphObjective(
    sai::knowledge::KnowledgeGraph& kg, double fp_cost, double fn_cost)
    : kg_(kg), fp_cost_(fp_cost), fn_cost_(fn_cost) {}

auto KnowledgeGraphObjective::SetThresholdParamIndices(
    int ng_threshold_idx, int warn_threshold_idx) -> void {
    ng_threshold_idx_ = ng_threshold_idx;
    warn_threshold_idx_ = warn_threshold_idx;
}

auto KnowledgeGraphObjective::Evaluate(const std::vector<double>& point,
                                       std::chrono::system_clock::time_point since)
    -> Result<double> {
    if (ng_threshold_idx_ >= 0 &&
        static_cast<std::size_t>(ng_threshold_idx_) < point.size()) {
        return EvaluateSimulated(point, since);
    }
    return EvaluateLegacy(since);
}

auto KnowledgeGraphObjective::EvaluateLegacy(
    std::chrono::system_clock::time_point since)
    -> Result<double> {
    auto nodes_result = kg_.FindNodesByType("GroundTruth");
    if (!nodes_result.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Tuning_ObjectiveEvalFailed,
            .message = "Failed to query GroundTruth nodes from knowledge graph",
            .source_location = std::source_location::current()});
    }

    auto since_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        since.time_since_epoch())
                        .count();

    std::int64_t fp = 0;
    std::int64_t fn = 0;
    std::int64_t tp = 0;
    std::int64_t tn = 0;

    for (const auto& node : nodes_result.value()) {
        // Extract timestamp and filter by since
        auto ts_it = node.properties.fields.find("timestamp");
        if (ts_it == node.properties.fields.end()) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} missing timestamp field, skipping", node.id);
            continue;
        }
        auto* ts_val = std::get_if<std::int64_t>(&ts_it->second);
        if (ts_val == nullptr) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} timestamp is not int64, skipping", node.id);
            continue;
        }
        if (*ts_val < since_us) continue;

        // Extract machine_verdict and human_label
        auto mv_it = node.properties.fields.find("machine_verdict");
        if (mv_it == node.properties.fields.end()) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} missing machine_verdict field, skipping", node.id);
            continue;
        }
        auto* mv_val = std::get_if<std::string>(&mv_it->second);
        if (mv_val == nullptr) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} machine_verdict is not string, skipping", node.id);
            continue;
        }

        // Ground truth: prefer human_label if available, else bootstrap
        // from machine_verdict (program self-calibration).
        bool ground_truth_ng = false;
        auto hl_it = node.properties.fields.find("human_label");
        if (hl_it != node.properties.fields.end()) {
            if (auto* hl_val = std::get_if<std::string>(&hl_it->second)) {
                ground_truth_ng = (*hl_val == "NG");
            }
        } else {
            // Bootstrap: use machine_verdict as self-supervised ground truth.
            // This enables the tuning loop to run without human intervention.
            ground_truth_ng = (*mv_val != "OK");
        }

        bool machine_ok = (*mv_val == "OK");

        if (!machine_ok && ground_truth_ng) {
            ++tp;
        } else if (machine_ok && !ground_truth_ng) {
            ++tn;
        } else if (!machine_ok && !ground_truth_ng) {
            ++fp;
        } else {
            ++fn;
        }
    }

    std::int64_t total = fp + fn + tp + tn;
    if (total == 0) return 0.0;

    return fp_cost_ * static_cast<double>(fp) / static_cast<double>(total) +
           fn_cost_ * static_cast<double>(fn) / static_cast<double>(total);
}

auto KnowledgeGraphObjective::EvaluateSimulated(
    const std::vector<double>& point,
    std::chrono::system_clock::time_point since)
    -> Result<double> {
    auto nodes_result = kg_.FindNodesByType("GroundTruth");
    if (!nodes_result.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Tuning_ObjectiveEvalFailed,
            .message = "Failed to query GroundTruth nodes from knowledge graph",
            .source_location = std::source_location::current()});
    }

    auto since_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        since.time_since_epoch())
                        .count();

    double ng_threshold = point[static_cast<std::size_t>(ng_threshold_idx_)];
    std::int64_t fp = 0;
    std::int64_t fn = 0;
    std::int64_t tp = 0;
    std::int64_t tn = 0;

    for (const auto& node : nodes_result.value()) {
        // Timestamp filter
        auto ts_it = node.properties.fields.find("timestamp");
        if (ts_it == node.properties.fields.end()) continue;
        auto* ts_val = std::get_if<std::int64_t>(&ts_it->second);
        if (ts_val == nullptr) continue;
        if (*ts_val < since_us) continue;

        // Ground truth: prefer human_label, else bootstrap from machine_verdict
        bool ground_truth_ng = false;
        auto hl_it = node.properties.fields.find("human_label");
        if (hl_it != node.properties.fields.end()) {
            if (auto* hl_val = std::get_if<std::string>(&hl_it->second)) {
                ground_truth_ng = (*hl_val == "NG");
            }
        } else {
            // Self-supervised bootstrap: use stored machine_verdict
            auto mv_it = node.properties.fields.find("machine_verdict");
            if (mv_it == node.properties.fields.end()) continue;
            auto* mv_val = std::get_if<std::string>(&mv_it->second);
            if (mv_val == nullptr) continue;
            ground_truth_ng = (*mv_val != "OK");
        }

        // Simulate machine verdict from detection_score + candidate threshold
        bool machine_ng = false;
        bool has_score = false;
        auto ds_it = node.properties.fields.find("detection_score");
        if (ds_it != node.properties.fields.end()) {
            if (auto* ds_val = std::get_if<double>(&ds_it->second)) {
                machine_ng = (*ds_val > ng_threshold);
                has_score = true;
            }
        }

        if (!has_score) {
            // Fallback to stored machine_verdict when detection_score is absent
            auto mv_it = node.properties.fields.find("machine_verdict");
            if (mv_it == node.properties.fields.end()) continue;
            auto* mv_val = std::get_if<std::string>(&mv_it->second);
            if (mv_val == nullptr) continue;
            machine_ng = (*mv_val != "OK");
        }

        // Classify
        if (machine_ng && ground_truth_ng)       ++tp;
        else if (!machine_ng && !ground_truth_ng) ++tn;
        else if (machine_ng && !ground_truth_ng)  ++fp;
        else /* !machine_ng && ground_truth_ng */ ++fn;
    }

    std::int64_t total = fp + fn + tp + tn;
    if (total == 0) return 0.0;

    return fp_cost_ * static_cast<double>(fp) / static_cast<double>(total) +
           fn_cost_ * static_cast<double>(fn) / static_cast<double>(total);
}

}  // namespace sai::tuning
