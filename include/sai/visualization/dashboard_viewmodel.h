#pragma once

#include <QObject>
#include <QAbstractListModel>
#include <QString>
#include <QVariantList>
#include <deque>
#include <string>
#include <string>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <map>

namespace sai::visualization {

struct FrameSummary {
    int frame_id{0};
    std::string verdict;
    std::string severity;
    std::chrono::system_clock::time_point timestamp;
    std::vector<std::string> defect_types;
};

class DefectTypeModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles { NameRole = Qt::UserRole + 1, CountRole, PctRole };
    explicit DefectTypeModel(QObject* parent = nullptr);
    auto rowCount(const QModelIndex& parent = QModelIndex()) const -> int override;
    auto data(const QModelIndex& index, int role = Qt::DisplayRole) const -> QVariant override;
    auto roleNames() const -> QHash<int, QByteArray> override;
    void RebuildFrom(const std::map<std::string, int>& type_counts, int total);
private:
    struct Row { QString name; int count; double pct; };
    std::vector<Row> rows_;
};

class DashboardViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(int totalFrames READ totalFrames NOTIFY summaryChanged)
    Q_PROPERTY(int okFrames READ okFrames NOTIFY summaryChanged)
    Q_PROPERTY(int ngFrames READ ngFrames NOTIFY summaryChanged)
    Q_PROPERTY(int warnFrames READ warnFrames NOTIFY summaryChanged)
    Q_PROPERTY(double ppm READ ppm NOTIFY summaryChanged)
    Q_PROPERTY(QObject* defectTypeModel READ defectTypeModel CONSTANT)
    Q_PROPERTY(QVariantList ppmTrend READ ppmTrend NOTIFY summaryChanged)
    Q_PROPERTY(QVariantList latencyTrend READ latencyTrend NOTIFY summaryChanged)
public:
    explicit DashboardViewModel(QObject* parent = nullptr);

    auto totalFrames() const -> int { return total_frames_.load(); }
    auto okFrames() const -> int { return ok_frames_.load(); }
    auto ngFrames() const -> int { return ng_frames_.load(); }
    auto warnFrames() const -> int { return warn_frames_.load(); }
    auto ppm() const -> double;
    auto defectTypeModel() -> QObject* { return defect_type_model_; }
    auto ppmTrend() const -> QVariantList { return ppm_trend_; }
    auto latencyTrend() const -> QVariantList { return latency_trend_; }

    void AppendFrameSummary(FrameSummary summary);

    static constexpr int kMaxFrameSummaries = 1000;

signals:
    void summaryChanged();

private:
    void RebuildTrends();

    std::deque<FrameSummary> frame_summaries_;
    mutable std::mutex deque_mutex_;

    std::atomic<int> total_frames_{0};
    std::atomic<int> ok_frames_{0};
    std::atomic<int> ng_frames_{0};
    std::atomic<int> warn_frames_{0};

    std::map<std::string, int> defect_type_counts_;

    DefectTypeModel* defect_type_model_;
    QVariantList ppm_trend_;
    QVariantList latency_trend_;
};

}  // namespace sai::visualization
