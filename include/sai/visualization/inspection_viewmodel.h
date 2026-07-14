#pragma once

#include <QObject>
#include <QAbstractListModel>
#include <QString>
#include <QRectF>
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>
#include <atomic>

namespace sai::reasoner {
struct ReasoningResult;
struct EvidenceItem;
}  // namespace sai::reasoner

#include "sai/detection/detection_result.h"
#include "sai/reasoner/evidence_collector.h"

namespace sai::visualization {

/// Single defect item exposed to QML (label, severity, confidence, bounding box).
class DefectItem : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString label READ label CONSTANT)
    Q_PROPERTY(QString severity READ severity CONSTANT)
    Q_PROPERTY(double confidence READ confidence CONSTANT)
    Q_PROPERTY(QRectF boundingBox READ boundingBox CONSTANT)
public:
    explicit DefectItem(QString label, QString severity, double confidence, QRectF bbox,
                        QObject* parent = nullptr);
    auto label() const -> QString { return label_; }
    auto severity() const -> QString { return severity_; }
    auto confidence() const -> double { return confidence_; }
    auto boundingBox() const -> QRectF { return bounding_box_; }
private:
    QString label_;
    QString severity_;
    double confidence_;
    QRectF bounding_box_;
};

/// QAbstractListModel holding the current frame's defects (from RegionProposals).
class DefectModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles {
        LabelRole = Qt::UserRole + 1,
        SeverityRole,
        ConfidenceRole,
        BoundingBoxRole
    };
    explicit DefectModel(QObject* parent = nullptr);
    auto rowCount(const QModelIndex& parent = QModelIndex()) const -> int override;
    auto data(const QModelIndex& index, int role = Qt::DisplayRole) const -> QVariant override;
    auto roleNames() const -> QHash<int, QByteArray> override;

    /// Update from detection RegionProposals (the closest equivalent to a "defect"
    /// in the current type system — no labeled Defect struct exists yet).
    /// Must be called from the model's owning thread; cross-thread callers should
    /// use QMetaObject::invokeMethod with Qt::QueuedConnection.
    Q_INVOKABLE void UpdateDefects(const std::vector<sai::detection::RegionProposal>& regions);

private:
    std::vector<std::unique_ptr<DefectItem>> defects_;
};

/// QAbstractListModel holding the current frame's evidence chain.
class EvidenceModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles {
        KeyRole = Qt::UserRole + 1,
        ValueRole,
        SourceRole,
        TraceRole
    };
    explicit EvidenceModel(QObject* parent = nullptr);
    auto rowCount(const QModelIndex& parent = QModelIndex()) const -> int override;
    auto data(const QModelIndex& index, int role = Qt::DisplayRole) const -> QVariant override;
    auto roleNames() const -> QHash<int, QByteArray> override;

    /// Must be called from the model's owning thread; cross-thread callers should
    /// use QMetaObject::invokeMethod with Qt::QueuedConnection.
    Q_INVOKABLE void UpdateEvidence(const std::vector<sai::reasoner::EvidenceItem>& evidence);

private:
    struct EvidenceRow {
        QString key, value, source, trace;
    };
    std::vector<EvidenceRow> rows_;
};

/// QML-facing viewmodel for per-frame inspection results.
/// UpdateResult() is designed to be called from the Export worker thread via
/// ResultCallback.  String fields are protected by a shared_mutex; frame_id
/// and confidence use std::atomic so they are always safe to read without
/// blocking the QML main thread.
class InspectionViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString verdict READ verdict NOTIFY resultChanged)
    Q_PROPERTY(QString severity READ severity NOTIFY resultChanged)
    Q_PROPERTY(QString recommendation READ recommendation NOTIFY resultChanged)
    Q_PROPERTY(double confidence READ confidence NOTIFY resultChanged)
    Q_PROPERTY(int frameId READ frameId NOTIFY resultChanged)
    Q_PROPERTY(QObject* defectModel READ defectModel CONSTANT)
    Q_PROPERTY(QObject* evidenceModel READ evidenceModel CONSTANT)
public:
    explicit InspectionViewModel(QObject* parent = nullptr);

    auto verdict() const -> QString;
    auto severity() const -> QString;
    auto recommendation() const -> QString;
    auto confidence() const -> double { return confidence_; }
    auto frameId() const -> int { return frame_id_; }
    auto defectModel() -> QObject* { return defect_model_; }
    auto evidenceModel() -> QObject* { return evidence_model_; }

    /// Called from ResultCallback (Export worker thread).  Acquires a
    /// unique_lock on data_mutex_ for the string fields; frame_id and
    /// confidence are written atomically.
    void UpdateResult(int frame_id, const sai::reasoner::ReasoningResult& result);

signals:
    void resultChanged();

private:
    DefectModel* defect_model_;
    EvidenceModel* evidence_model_;

    std::atomic<int> frame_id_{0};
    std::string verdict_;
    std::string severity_;
    std::string recommendation_;
    std::atomic<double> confidence_{0.0};
    mutable std::shared_mutex data_mutex_;
};

}  // namespace sai::visualization
