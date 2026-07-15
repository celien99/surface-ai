#include <sai/tuning/tuning_objective.h>

#include <cstdint>
#include <string>

#include <sai/infra/logger.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::tuning {

KnowledgeGraphObjective::KnowledgeGraphObjective(
    sai::knowledge::KnowledgeGraph& kg, double fp_cost, double fn_cost)
    : kg_(kg), fp_cost_(fp_cost), fn_cost_(fn_cost) {}

auto KnowledgeGraphObjective::Evaluate(const std::vector<double>& /*point*/,
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

        auto hl_it = node.properties.fields.find("human_label");
        if (hl_it == node.properties.fields.end()) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} missing human_label field, skipping", node.id);
            continue;
        }
        auto* hl_val = std::get_if<std::string>(&hl_it->second);
        if (hl_val == nullptr) {
            sai::infra::Logger::Get("tuning").Log(
                sai::infra::LogLevel::Warning,
                "GroundTruth node {} human_label is not string, skipping", node.id);
            continue;
        }

        const auto& machine = *mv_val;
        const auto& human = *hl_val;
        bool machine_ok = (machine == "OK");
        bool human_ng = (human == "NG");

        if (!machine_ok && human_ng) {
            ++tp;  // machine flagged, human confirms defect
        } else if (machine_ok && !human_ng) {
            ++tn;  // both agree OK
        } else if (!machine_ok && !human_ng) {
            ++fp;  // machine flagged, but human says OK (false alarm)
        } else {  // machine_ok && human_ng
            ++fn;  // machine missed, but human says NG (missed defect)
        }
    }

    std::int64_t total = fp + fn + tp + tn;
    if (total == 0) return 0.0;

    return fp_cost_ * static_cast<double>(fp) / static_cast<double>(total) +
           fn_cost_ * static_cast<double>(fn) / static_cast<double>(total);
}

}  // namespace sai::tuning
