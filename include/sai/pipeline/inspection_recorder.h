// inspection_recorder.h — 将带 detection_score 的检测记录异步批量写入 KnowledgeGraph
// Batch T2: 修复贝叶斯优化仿真数据缺失问题。
// Batch T3: 异步缓冲写入——热路径无锁 push，后台线程批量 flush。
//
// 设计决策：
// - 热路径 WriteRecord 仅做无锁 push 到环形缓冲，不阻塞调用者。
// - 后台 flush 线程每 100ms 或缓冲满 500 条时批量写入 SQLite，使用
//   KnowledgeGraph::InsertNodesBatch 在单个事务内完成。
// - 析构时 drain 剩余记录，保证不丢数据。
// - 缓冲满时 drop_oldest（best-effort 语义），tuning 数据丢失可接受。
// - human_label 字段保留供人工标注流程后续填充。

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

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

    ~InspectionRecorder() noexcept;

    InspectionRecorder(const InspectionRecorder&) = delete;
    auto operator=(const InspectionRecorder&) -> InspectionRecorder& = delete;
    InspectionRecorder(InspectionRecorder&&) = delete;
    auto operator=(InspectionRecorder&&) -> InspectionRecorder& = delete;

    // 异步写入一条待标注的检测记录（GroundTruth 节点）。
    //
    // 热路径 safe：仅 push 到无锁环形缓冲，不阻塞调用者。
    // detection_score: 原始异常分数（来自 DetectionResult::image_level_score）
    // machine_verdict: 当前阈值下的机器判定（"OK" / "NG"）
    // surface_id: 被检测的面区标识
    // timestamp_us: 检测时刻（epoch 微秒）；0 表示调用方不关心精确时刻
    //
    // 缓冲满时静默丢弃最旧记录（best-effort）。
    auto WriteRecord(
        double detection_score,
        std::string_view machine_verdict,
        std::string_view surface_id,
        std::int64_t timestamp_us = 0) noexcept -> void;

    // 控制是否启用记录（默认 false，避免生产环境意外写入）。
    auto SetEnabled(bool enabled) noexcept -> void;
    [[nodiscard]] auto IsEnabled() const noexcept -> bool { return enabled_; }

    // 立即 flush 所有缓冲记录（用于优雅停机）。
    auto Flush() noexcept -> void;

private:
    struct PendingRecord {
        double detection_score;
        std::string machine_verdict;
        std::string surface_id;
        std::int64_t timestamp_us;
    };

    // Flush 线程主循环
    void FlushLoop(std::stop_token stop_token) noexcept;

    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    std::atomic<bool> enabled_{false};

    // 双缓冲：write_buffer_ 接收热路径 push，flush 线程 swap 到 flush_buffer_ 后写入
    std::mutex buffer_mutex_;
    std::vector<PendingRecord> write_buffer_;
    std::vector<PendingRecord> flush_buffer_;

    std::jthread flush_thread_;
    std::atomic<bool> started_{false};

    static constexpr std::size_t kMaxBufferSize = 512;
    static constexpr std::chrono::milliseconds kFlushInterval{100};
};

}  // namespace sai::pipeline
