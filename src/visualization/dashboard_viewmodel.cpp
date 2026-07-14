#include "sai/visualization/dashboard_viewmodel.h"
#include <algorithm>
#include <cmath>

namespace sai::visualization {

DefectTypeModel::DefectTypeModel(QObject* parent) : QAbstractListModel(parent) {}

auto DefectTypeModel::rowCount(const QModelIndex& parent) const -> int {
    if (parent.isValid()) return 0;
    return static_cast<int>(rows_.size());
}

auto DefectTypeModel::data(const QModelIndex& index, int role) const -> QVariant {
    if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(rows_.size()))
        return {};
    const auto& row = rows_[index.row()];
    switch (role) {
        case NameRole:  return row.name;
        case CountRole: return row.count;
        case PctRole:   return row.pct;
        case Qt::DisplayRole: return row.name;
        default: return {};
    }
}

auto DefectTypeModel::roleNames() const -> QHash<int, QByteArray> {
    return {{NameRole, "defectName"}, {CountRole, "count"}, {PctRole, "percentage"}};
}

void DefectTypeModel::RebuildFrom(const std::map<std::string, int>& type_counts, int total) {
    beginResetModel();
    rows_.clear();
    for (const auto& [name, count] : type_counts) {
        rows_.push_back({QString::fromStdString(name), count,
                         total > 0 ? (100.0 * count / total) : 0.0});
    }
    // Sort by count descending
    std::sort(rows_.begin(), rows_.end(),
              [](const Row& a, const Row& b) { return a.count > b.count; });
    endResetModel();
}

DashboardViewModel::DashboardViewModel(QObject* parent)
    : QObject(parent), defect_type_model_(new DefectTypeModel(this)) {}

auto DashboardViewModel::ppm() const -> double {
    auto total = total_frames_.load();
    if (total == 0) return 0.0;
    return 1'000'000.0 * (ng_frames_.load() + warn_frames_.load()) / total;
}

void DashboardViewModel::AppendFrameSummary(FrameSummary summary) {
    auto ng = summary.verdict == "NG" ? 1 : 0;
    auto ok = summary.verdict == "OK" ? 1 : 0;
    auto warn = summary.verdict == "WARN" || summary.verdict == "UNCERTAIN" ? 1 : 0;

    {
        std::lock_guard lock(deque_mutex_);
        for (const auto& dt : summary.defect_types) {
            defect_type_counts_[dt]++;
        }
        frame_summaries_.push_back(std::move(summary));
        if (frame_summaries_.size() > static_cast<size_t>(kMaxFrameSummaries)) {
            frame_summaries_.pop_front();
        }
    }

    total_frames_.fetch_add(1);
    ok_frames_.fetch_add(ok);
    ng_frames_.fetch_add(ng);
    warn_frames_.fetch_add(warn);

    defect_type_model_->RebuildFrom(defect_type_counts_, total_frames_.load());

    RebuildTrends();
    emit summaryChanged();
}

void DashboardViewModel::RebuildTrends() {
    std::lock_guard lock(deque_mutex_);
    ppm_trend_.clear();
    latency_trend_.clear();
    int ok_count = 0, ng_count = 0;
    for (const auto& fs : frame_summaries_) {
        if (fs.verdict == "OK") ok_count++;
        else if (fs.verdict == "NG") ng_count++;
        int total = ok_count + ng_count;
        double current_ppm = total > 0 ? 1'000'000.0 * ng_count / total : 0.0;
        QVariantMap point;
        point["x"] = static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                fs.timestamp.time_since_epoch()).count());
        point["y"] = current_ppm;
        ppm_trend_.append(point);
    }
}

}  // namespace sai::visualization
