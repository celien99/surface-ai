#include <sai/pipeline/inspection_recorder.h>

#include <source_location>
#include <string>

#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::pipeline {

auto InspectionRecorder::WriteRecord(
    double detection_score,
    std::string_view machine_verdict,
    std::string_view surface_id,
    std::int64_t timestamp_us) const noexcept
    -> Result<std::int64_t> {
    if (!enabled_ || !kg_) {
        return 0;  // silently no-op when disabled
    }

    // Resolve timestamp if caller didn't supply one
    if (timestamp_us == 0) {
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    }

    sai::knowledge::KnowledgeRecord props;
    props.fields["detection_score"] = detection_score;
    props.fields["machine_verdict"] = std::string(machine_verdict);
    props.fields["timestamp"] = timestamp_us;

    if (!surface_id.empty()) {
        props.fields["surface_id"] = std::string(surface_id);
    }

    // human_label 未设置——由人工标注流程后续填充。
    // EvaluateSimulated 在 human_label 缺失时跳过该条记录。

    return kg_->InsertNode("GroundTruth", std::move(props));
}

}  // namespace sai::pipeline
