#include <gtest/gtest.h>
#include <QGuiApplication>
#include <QCoreApplication>
#include <thread>
#include <vector>

#include "sai/visualization/inspection_viewmodel.h"
#include "sai/detection/detection_result.h"
#include "sai/reasoner/reasoner.h"
#include "sai/reasoner/evidence_collector.h"

// ---------------------------------------------------------------------------
// DefectModel tests
// ---------------------------------------------------------------------------
TEST(DefectModelTest, UpdateDefectsReturnsCorrectRowCount) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DefectModel model;

    std::vector<sai::detection::RegionProposal> regions;
    for (int i = 0; i < 3; ++i) {
        sai::detection::RegionProposal r;
        r.bounding_box = {static_cast<size_t>(i * 10), 0, 100, 200};
        r.max_anomaly_score = 0.5F + 0.1F * i;
        r.mean_anomaly_score = 0.3F;
        r.area_pixels = 1000;
        regions.push_back(r);
    }

    model.UpdateDefects(regions);
    EXPECT_EQ(model.rowCount(), 3);
}

TEST(DefectModelTest, DataReturnsCorrectFields) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DefectModel model;

    sai::detection::RegionProposal r;
    r.bounding_box = {10, 20, 100, 200};
    r.max_anomaly_score = 0.85F;
    std::vector<sai::detection::RegionProposal> regions = {r};

    model.UpdateDefects(regions);

    QModelIndex idx = model.index(0);
    EXPECT_EQ(model.data(idx, sai::visualization::DefectModel::LabelRole).toString().toStdString(),
              "region_0");
    EXPECT_TRUE(model.data(idx, sai::visualization::DefectModel::SeverityRole).toString().toStdString() == "HIGH");
    EXPECT_NEAR(model.data(idx, sai::visualization::DefectModel::ConfidenceRole).toDouble(), 0.85, 0.001);
    QRectF bbox = model.data(idx, sai::visualization::DefectModel::BoundingBoxRole).toRectF();
    EXPECT_NEAR(bbox.x(), 10.0, 0.01);
    EXPECT_NEAR(bbox.y(), 20.0, 0.01);
    EXPECT_NEAR(bbox.width(), 100.0, 0.01);
    EXPECT_NEAR(bbox.height(), 200.0, 0.01);
}

// ---------------------------------------------------------------------------
// EvidenceModel tests
// ---------------------------------------------------------------------------
TEST(EvidenceModelTest, UpdateEvidenceReturnsCorrectRowCount) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::EvidenceModel model;

    std::vector<sai::reasoner::EvidenceItem> evidence;
    for (int i = 0; i < 2; ++i) {
        sai::reasoner::EvidenceItem item;
        item.key = "rule_" + std::to_string(i);
        item.value = sai::rule::Value::Of(0.95);
        item.source.description = "computed";
        item.trace_id = "trace_" + std::to_string(i);
        evidence.push_back(std::move(item));
    }

    model.UpdateEvidence(evidence);
    EXPECT_EQ(model.rowCount(), 2);
}

TEST(EvidenceModelTest, DataReturnsCorrectFields) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::EvidenceModel model;

    sai::reasoner::EvidenceItem item;
    item.key = "brightness_check";
    item.value = sai::rule::Value::Of(std::string("FAIL"));
    item.source.description = "direct_measurement";
    item.trace_id = "t001";
    std::vector<sai::reasoner::EvidenceItem> evidence = {item};

    model.UpdateEvidence(evidence);

    QModelIndex idx = model.index(0);
    EXPECT_EQ(model.data(idx, sai::visualization::EvidenceModel::KeyRole).toString().toStdString(),
              "brightness_check");
    EXPECT_EQ(model.data(idx, sai::visualization::EvidenceModel::ValueRole).toString().toStdString(),
              "FAIL");
    EXPECT_EQ(model.data(idx, sai::visualization::EvidenceModel::SourceRole).toString().toStdString(),
              "direct_measurement");
    EXPECT_EQ(model.data(idx, sai::visualization::EvidenceModel::TraceRole).toString().toStdString(),
              "t001");
}

// ---------------------------------------------------------------------------
// InspectionViewModel tests
// ---------------------------------------------------------------------------
TEST(InspectionViewModelTest, UpdateResultEmitsSignal) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::InspectionViewModel vm;

    int signal_count = 0;
    QObject::connect(&vm, &sai::visualization::InspectionViewModel::resultChanged,
                     [&signal_count] { ++signal_count; });

    sai::reasoner::ReasoningResult rr;
    rr.verdict = "NG";
    rr.severity = 0.85;
    rr.recommendation = "Stop line immediately";
    rr.confidence = 0.92;

    vm.UpdateResult(42, rr);

    EXPECT_EQ(signal_count, 1);
    EXPECT_EQ(vm.frameId(), 42);
    EXPECT_NEAR(vm.confidence(), 0.92, 0.001);
    EXPECT_EQ(vm.verdict().toStdString(), "NG");
    EXPECT_EQ(vm.recommendation().toStdString(), "Stop line immediately");
}

TEST(InspectionViewModelTest, UpdateResultPopulatesEvidenceModel) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::InspectionViewModel vm;

    sai::reasoner::ReasoningResult rr;
    rr.verdict = "OK";
    rr.severity = 0.0;
    rr.recommendation = "Pass";
    rr.confidence = 0.98;

    sai::reasoner::EvidenceItem item;
    item.key = "stitching_check";
    item.value = sai::rule::Value::Of(true);
    item.source.description = "boolean_rule";
    item.trace_id = "t_stitch";
    rr.evidence.push_back(item);

    vm.UpdateResult(1, rr);

    // Process queued invocations (UpdateEvidence is marshalled via
    // QMetaObject::invokeMethod with Qt::QueuedConnection for thread safety).
    QCoreApplication::processEvents();

    auto* em = qobject_cast<sai::visualization::EvidenceModel*>(vm.evidenceModel());
    ASSERT_NE(em, nullptr);
    EXPECT_EQ(em->rowCount(), 1);
}

TEST(InspectionViewModelTest, ThreadSafetySmoke) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::InspectionViewModel vm;

    auto worker = [&vm](int base_id, const std::string& verdict, double severity) {
        for (int i = 0; i < 100; ++i) {
            sai::reasoner::ReasoningResult rr;
            rr.verdict = verdict;
            rr.severity = severity;
            rr.recommendation = "auto";
            rr.confidence = 0.5;
            vm.UpdateResult(base_id + i, rr);
        }
    };

    std::thread t1(worker, 1000, "NG", 0.9);
    std::thread t2(worker, 2000, "OK", 0.1);

    t1.join();
    t2.join();

    // No crash = pass.  Verify the viewmodel is still usable after the storm.
    EXPECT_GE(vm.frameId(), 0);
}
