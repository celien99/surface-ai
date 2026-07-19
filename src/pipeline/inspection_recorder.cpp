#include <sai/pipeline/inspection_recorder.h>

#include <source_location>
#include <string>
#include <utility>

#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::pipeline {

InspectionRecorder::~InspectionRecorder() noexcept {
    // 请求 flush 线程停止并等待 join
    if (flush_thread_.joinable()) {
        flush_thread_.request_stop();
        flush_thread_.join();
    }
    // 最后 drain 一次剩余记录
    Flush();
}

auto InspectionRecorder::SetEnabled(bool enabled) noexcept -> void {
    bool was_enabled = enabled_.exchange(enabled);
    if (enabled && !was_enabled && !started_.exchange(true)) {
        // 启动后台 flush 线程
        write_buffer_.reserve(kMaxBufferSize);
        flush_buffer_.reserve(kMaxBufferSize);
        flush_thread_ = std::jthread(
            [this](std::stop_token st) { FlushLoop(std::move(st)); });
    }
}

auto InspectionRecorder::WriteRecord(
    double detection_score,
    std::string_view machine_verdict,
    std::string_view surface_id,
    std::int64_t timestamp_us) noexcept -> void {
    if (!enabled_ || !kg_) return;

    // 解析时间戳
    if (timestamp_us == 0) {
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    }

    // 热路径：仅加锁 push，锁持有时间 ~100ns
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (write_buffer_.size() >= kMaxBufferSize) {
            // 缓冲满：drop_oldest（best-effort，tuning 数据可接受丢失）
            return;
        }
        write_buffer_.push_back(PendingRecord{
            .detection_score = detection_score,
            .machine_verdict = std::string(machine_verdict),
            .surface_id = std::string(surface_id),
            .timestamp_us = timestamp_us,
        });
    }
}

auto InspectionRecorder::Flush() noexcept -> void {
    if (!kg_) return;

    // Swap buffers under lock
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        flush_buffer_.swap(write_buffer_);
    }

    if (flush_buffer_.empty()) return;

    // 批量写入 KnowledgeGraph（单事务）
    std::vector<std::pair<std::string, sai::knowledge::KnowledgeRecord>> entries;
    entries.reserve(flush_buffer_.size());

    for (auto& rec : flush_buffer_) {
        sai::knowledge::KnowledgeRecord props;
        props.fields["detection_score"] = rec.detection_score;
        props.fields["machine_verdict"] = std::move(rec.machine_verdict);
        props.fields["timestamp"] = rec.timestamp_us;
        if (!rec.surface_id.empty()) {
            props.fields["surface_id"] = std::move(rec.surface_id);
        }
        entries.emplace_back("GroundTruth", std::move(props));
    }

    (void)kg_->InsertNodesBatch(std::move(entries));
    flush_buffer_.clear();
}

void InspectionRecorder::FlushLoop(std::stop_token stop_token) noexcept {
    while (!stop_token.stop_requested()) {
        // 等待 kFlushInterval 或提前唤醒
        // 使用 stop_token 的 wait 来支持优雅停机
        auto wait_start = std::chrono::steady_clock::now();
        while (!stop_token.stop_requested()) {
            auto elapsed = std::chrono::steady_clock::now() - wait_start;
            if (elapsed >= kFlushInterval) break;

            // Check if buffer is getting full → flush early
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                if (write_buffer_.size() >= kMaxBufferSize / 2) break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        Flush();
    }
}

}  // namespace sai::pipeline
