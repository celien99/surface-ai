// inspection_recorder.h — 将带 detection_score 的检测记录写入 KnowledgeGraph
// Batch T2: 修复贝叶斯优化仿真数据缺失问题。
//
// 设计决策：
// - InspectionRecorder 使用已持有的 KnowledgeGraph 句柄，不获取所有权。
//   调用者（RuleEvalStage 或 Pipeline）负责管理 KG 生命周期。
// - WritePendingRecord 写入 GroundTruth 节点，包含 detection_score、
//   machine_verdict、timestamp 和 surface_id。
//   human_label 字段保留供人工标注流程后续填充。
// - detection_score 是 DetectionResult 中的原始异常分数（image_level_score），
//   而非二值化后的 OK/NG。TuningObjective::EvaluateSimulated 使用候选阈值
//   与该分数比较来重新计算 FP/FN/TP/TN。
// - 记录插入失败不阻塞管线（best-effort 语义）。

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <sai/core/error.h>

namespace sai::knowledge {
class KnowledgeGraph;
}  // namespace sai::knowledge

namespace sai::pipeline {

class InspectionRecorder final {
public:
    explicit InspectionRecorder(
        std::shared_ptr<sai::knowledge::KnowledgeGraph> kg) noexcept
        : kg_(std::move(kg)) {}

    // 写入一条待标注的检测记录（GroundTruth 节点）。
    //
    // detection_score: 原始异常分数（来自 DetectionResult::image_level_score）
    // machine_verdict: 当前阈值下的机器判定（"OK" / "NG"）
    // surface_id: 被检测的面区标识
    // timestamp_us: 检测时刻（epoch 微秒）；0 表示调用方不关心精确时刻
    //
    // 返回新节点的 ID（可用于后续人工标注流程更新 human_label），
    // 或错误信息（不阻塞管线）。
    [[nodiscard]] auto WriteRecord(
        double detection_score,
        std::string_view machine_verdict,
        std::string_view surface_id,
        std::int64_t timestamp_us = 0) const noexcept
        -> Result<std::int64_t>;

    // 控制是否启用记录（默认 false，避免生产环境意外写入）。
    auto SetEnabled(bool enabled) noexcept -> void { enabled_ = enabled; }
    [[nodiscard]] auto IsEnabled() const noexcept -> bool { return enabled_; }

private:
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    bool enabled_ = false;
};

}  // namespace sai::pipeline
