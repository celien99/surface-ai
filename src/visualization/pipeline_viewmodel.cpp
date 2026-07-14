#include "sai/visualization/pipeline_viewmodel.h"
#include "sai/pipeline/pipeline.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace sai::visualization {

StageMetricsObject::StageMetricsObject(QString stageId, QString stageType, QObject* parent)
    : QObject(parent), stage_id_(std::move(stageId)), stage_type_(std::move(stageType)) {}

void StageMetricsObject::UpdateFromStageMetrics(const sai::pipeline::StageMetrics& m) {
    frames_processed_ = static_cast<int>(m.frames_processed.load());
    frames_failed_ = static_cast<int>(m.frames_failed.load());
    frames_dropped_ = static_cast<int>(m.frames_dropped.load());
    queue_depth_ = static_cast<int>(m.queue_depth());
    avg_latency_ms_ = m.avg_latency_us / 1000.0;
    p99_latency_ms_ = m.p99_latency_us / 1000.0;
    emit metricsChanged();
}

PipelineViewModel::PipelineViewModel(QObject* parent) : QObject(parent) {
    QObject::connect(&refresh_timer_, &QTimer::timeout, this, &PipelineViewModel::Refresh);
}

void PipelineViewModel::BindToPipeline(const sai::pipeline::Pipeline* pipeline) {
    pipeline_ = pipeline;
    // Build StageMetricsObject list once on bind
    if (pipeline_) {
        pipeline_status_ = "Running";
        const auto metrics = pipeline_->Metrics();
        for (const auto& m : metrics) {
            auto* obj = new StageMetricsObject(
                QString::fromStdString(m.stage_id),
                QString::fromStdString(std::to_string(static_cast<int>(m.type))),
                this);
            obj->UpdateFromStageMetrics(m);
            stage_metrics_objects_.append(obj);
        }
    }
    emit statusChanged();
    emit stageMetricsChanged();
}

void PipelineViewModel::StartRefresh(int intervalMs) {
    last_frames_processed_ = 0;
    refresh_timer_.start(intervalMs);
}

void PipelineViewModel::StopRefresh() {
    refresh_timer_.stop();
}

void PipelineViewModel::Refresh() {
    if (!pipeline_) {
        pipeline_status_ = "Stopped";
        emit statusChanged();
        return;
    }

    const auto metrics = pipeline_->Metrics();
    if (metrics.empty()) return;

    int total_processed = 0;
    int total_failed = 0;
    int total_dropped = 0;

    for (size_t i = 0; i < metrics.size() && i < static_cast<size_t>(stage_metrics_objects_.size()); ++i) {
        const auto& m = metrics[i];
        auto* obj = qobject_cast<StageMetricsObject*>(stage_metrics_objects_[i]);
        if (obj) obj->UpdateFromStageMetrics(m);
        total_processed += static_cast<int>(m.frames_processed.load());
        total_failed += static_cast<int>(m.frames_failed.load());
        total_dropped += static_cast<int>(m.frames_dropped.load());
    }

    // Calculate FPS from delta since last refresh
    int delta = total_processed - last_frames_processed_;
    overall_fps_ = delta / (refresh_timer_.interval() / 1000.0);
    last_frames_processed_ = total_processed;

    total_frames_processed_ = total_processed;
    total_frames_failed_ = total_failed;
    in_flight_frames_ = total_processed - total_failed - total_dropped;

    pipeline_status_ = "Running";
    emit statusChanged();
    emit metricsChanged();
}

}  // namespace sai::visualization
