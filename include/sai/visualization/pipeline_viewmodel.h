#pragma once

#include <QObject>
#include <QTimer>
#include <QList>
#include <QString>
#include <string>

namespace sai::pipeline {
struct StageMetrics;
class Pipeline;
}  // namespace sai::pipeline

namespace sai::visualization {

/// QML-facing wrapper around a single stage's StageMetrics.
/// Updated by PipelineViewModel's refresh timer on the QML main thread.
class StageMetricsObject : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString stageId READ stageId CONSTANT)
    Q_PROPERTY(QString stageType READ stageType CONSTANT)
    Q_PROPERTY(int framesProcessed READ framesProcessed NOTIFY metricsChanged)
    Q_PROPERTY(int framesFailed READ framesFailed NOTIFY metricsChanged)
    Q_PROPERTY(int framesDropped READ framesDropped NOTIFY metricsChanged)
    Q_PROPERTY(int queueDepth READ queueDepth NOTIFY metricsChanged)
    Q_PROPERTY(double avgLatencyMs READ avgLatencyMs NOTIFY metricsChanged)
    Q_PROPERTY(double p99LatencyMs READ p99LatencyMs NOTIFY metricsChanged)
public:
    explicit StageMetricsObject(QString stageId, QString stageType, QObject* parent = nullptr);

    auto stageId() const -> QString { return stage_id_; }
    auto stageType() const -> QString { return stage_type_; }
    auto framesProcessed() const -> int { return frames_processed_; }
    auto framesFailed() const -> int { return frames_failed_; }
    auto framesDropped() const -> int { return frames_dropped_; }
    auto queueDepth() const -> int { return queue_depth_; }
    auto avgLatencyMs() const -> double { return avg_latency_ms_; }
    auto p99LatencyMs() const -> double { return p99_latency_ms_; }

    void UpdateFromStageMetrics(const sai::pipeline::StageMetrics& metrics);

signals:
    void metricsChanged();

private:
    QString stage_id_;
    QString stage_type_;
    int frames_processed_{0};
    int frames_failed_{0};
    int frames_dropped_{0};
    int queue_depth_{0};
    double avg_latency_ms_{0.0};
    double p99_latency_ms_{0.0};
};

/// QML-facing viewmodel for Pipeline status and per-stage metrics.
/// Binds to a Pipeline via raw pointer (does not own).
/// StartRefresh() begins a QTimer-driven polling loop on the QML main thread.
class PipelineViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString pipelineName READ pipelineName CONSTANT)
    Q_PROPERTY(QString pipelineStatus READ pipelineStatus NOTIFY statusChanged)
    Q_PROPERTY(double overallFps READ overallFps NOTIFY metricsChanged)
    Q_PROPERTY(int totalFramesProcessed READ totalFramesProcessed NOTIFY metricsChanged)
    Q_PROPERTY(int totalFramesFailed READ totalFramesFailed NOTIFY metricsChanged)
    Q_PROPERTY(int inFlightFrames READ inFlightFrames NOTIFY metricsChanged)
    Q_PROPERTY(QList<QObject*> stageMetrics READ stageMetrics NOTIFY stageMetricsChanged)
public:
    explicit PipelineViewModel(QObject* parent = nullptr);

    void BindToPipeline(const sai::pipeline::Pipeline* pipeline);

    auto pipelineName() const -> QString { return pipeline_name_; }
    auto pipelineStatus() const -> QString { return pipeline_status_; }
    auto overallFps() const -> double { return overall_fps_; }
    auto totalFramesProcessed() const -> int { return total_frames_processed_; }
    auto totalFramesFailed() const -> int { return total_frames_failed_; }
    auto inFlightFrames() const -> int { return in_flight_frames_; }
    auto stageMetrics() const -> QList<QObject*> { return stage_metrics_objects_; }

    void StartRefresh(int intervalMs = 33);
    void StopRefresh();

signals:
    void statusChanged();
    void metricsChanged();
    void stageMetricsChanged();

private:
    void Refresh();

    const sai::pipeline::Pipeline* pipeline_{nullptr};
    QString pipeline_name_;
    QString pipeline_status_{"Stopped"};
    double overall_fps_{0.0};
    int total_frames_processed_{0};
    int total_frames_failed_{0};
    int in_flight_frames_{0};
    QTimer refresh_timer_;
    QList<QObject*> stage_metrics_objects_;
    int last_frames_processed_{0};
};

}  // namespace sai::visualization
