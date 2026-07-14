#include "sai/visualization/inspection_viewmodel.h"

#include "sai/reasoner/reasoner.h"
#include "sai/reasoner/evidence_collector.h"
#include "sai/detection/detection_result.h"
#include "sai/rule/value.h"

#include <cmath>
#include <shared_mutex>

namespace sai::visualization {

// ---------------------------------------------------------------------------
// DefectItem
// ---------------------------------------------------------------------------
DefectItem::DefectItem(QString label, QString severity, double confidence, QRectF bbox,
                       QObject* parent)
    : QObject(parent)
    , label_(std::move(label))
    , severity_(std::move(severity))
    , confidence_(confidence)
    , bounding_box_(bbox) {}

// ---------------------------------------------------------------------------
// DefectModel
// ---------------------------------------------------------------------------
DefectModel::DefectModel(QObject* parent) : QAbstractListModel(parent) {}

int DefectModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return static_cast<int>(defects_.size());
}

QVariant DefectModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(defects_.size()))
        return {};

    const auto& d = defects_[static_cast<size_t>(index.row())];
    switch (role) {
    case LabelRole:        return d->label();
    case SeverityRole:     return d->severity();
    case ConfidenceRole:   return d->confidence();
    case BoundingBoxRole:  return d->boundingBox();
    case Qt::DisplayRole:  return d->label();
    default:               return {};
    }
}

QHash<int, QByteArray> DefectModel::roleNames() const {
    return {
        {LabelRole,       "label"},
        {SeverityRole,    "severity"},
        {ConfidenceRole,  "confidence"},
        {BoundingBoxRole, "boundingBox"}
    };
}

static QString SeverityFromScore(float score) {
    if (score > 0.7F) return QStringLiteral("HIGH");
    if (score > 0.3F) return QStringLiteral("MEDIUM");
    return QStringLiteral("LOW");
}

void DefectModel::UpdateDefects(const std::vector<sai::detection::RegionProposal>& regions) {
    beginResetModel();
    defects_.clear();
    defects_.reserve(regions.size());

    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& r = regions[i];
        auto label = QStringLiteral("region_%1").arg(static_cast<int>(i));
        auto severity = SeverityFromScore(r.max_anomaly_score);
        auto confidence = static_cast<double>(r.max_anomaly_score);
        QRectF bbox(static_cast<qreal>(r.bounding_box.x),
                    static_cast<qreal>(r.bounding_box.y),
                    static_cast<qreal>(r.bounding_box.width),
                    static_cast<qreal>(r.bounding_box.height));

        defects_.push_back(std::make_unique<DefectItem>(label, severity, confidence, bbox));
    }
    endResetModel();
}

// ---------------------------------------------------------------------------
// EvidenceModel
// ---------------------------------------------------------------------------
EvidenceModel::EvidenceModel(QObject* parent) : QAbstractListModel(parent) {}

int EvidenceModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return static_cast<int>(rows_.size());
}

static QString ValueToDisplayString(const sai::rule::Value& v) {
    if (auto s = v.AsString()) return QString::fromStdString(std::string(*s));
    if (auto d = v.AsDouble()) return QString::number(*d);
    if (auto b = v.AsBool()) return *b ? QStringLiteral("true") : QStringLiteral("false");
    if (auto* list = v.AsList()) return QStringLiteral("[%1 items]").arg(static_cast<int>(list->size()));
    return QStringLiteral("(null)");
}

QVariant EvidenceModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(rows_.size()))
        return {};

    const auto& row = rows_[static_cast<size_t>(index.row())];
    switch (role) {
    case KeyRole:    return row.key;
    case ValueRole:  return row.value;
    case SourceRole: return row.source;
    case TraceRole:  return row.trace;
    case Qt::DisplayRole: return row.key;
    default:         return {};
    }
}

QHash<int, QByteArray> EvidenceModel::roleNames() const {
    return {
        {KeyRole,    "evidenceKey"},
        {ValueRole,  "evidenceValue"},
        {SourceRole, "evidenceSource"},
        {TraceRole,  "evidenceTrace"}
    };
}

void EvidenceModel::UpdateEvidence(const std::vector<sai::reasoner::EvidenceItem>& evidence) {
    beginResetModel();
    rows_.clear();
    rows_.reserve(evidence.size());

    for (const auto& item : evidence) {
        EvidenceRow row;
        row.key    = QString::fromStdString(item.key);
        row.value  = ValueToDisplayString(item.value);
        row.source = QString::fromStdString(item.source.description);
        row.trace  = QString::fromStdString(item.trace_id);
        rows_.push_back(std::move(row));
    }
    endResetModel();
}

// ---------------------------------------------------------------------------
// InspectionViewModel
// ---------------------------------------------------------------------------
InspectionViewModel::InspectionViewModel(QObject* parent)
    : QObject(parent)
    , defect_model_(new DefectModel(this))
    , evidence_model_(new EvidenceModel(this)) {}

QString InspectionViewModel::verdict() const {
    std::shared_lock lock(data_mutex_);
    return QString::fromStdString(verdict_);
}

QString InspectionViewModel::severity() const {
    std::shared_lock lock(data_mutex_);
    return QString::fromStdString(severity_);
}

QString InspectionViewModel::recommendation() const {
    std::shared_lock lock(data_mutex_);
    return QString::fromStdString(recommendation_);
}

void InspectionViewModel::UpdateResult(int frame_id, const sai::reasoner::ReasoningResult& result) {
    frame_id_.store(frame_id, std::memory_order_release);
    confidence_.store(result.confidence, std::memory_order_release);

    {
        std::unique_lock lock(data_mutex_);
        verdict_        = result.verdict;
        severity_       = std::to_string(result.severity);  // double → string for QML display
        recommendation_ = result.recommendation;
    }

    // Marshal evidence update to the main thread — UpdateEvidence() calls
    // beginResetModel()/endResetModel() which must run on the owning thread.
    // DefectModel is NOT updated here because ReasoningResult carries no
    // region/defect data — regions live in DetectionResult which flows through
    // a separate pipeline stage.
    QMetaObject::invokeMethod(evidence_model_, "UpdateEvidence", Qt::QueuedConnection,
                              Q_ARG(std::vector<sai::reasoner::EvidenceItem>, result.evidence));

    emit resultChanged();
}

}  // namespace sai::visualization
