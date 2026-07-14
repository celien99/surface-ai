#include <gtest/gtest.h>
#include <QGuiApplication>

#include "sai/visualization/dashboard_viewmodel.h"

TEST(DashboardViewModelTest, AppendSummaryIncrementsTotalFrames) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;
    sai::visualization::FrameSummary fs;
    fs.frame_id = 1;
    fs.verdict = "OK";
    fs.timestamp = std::chrono::system_clock::now();

    vm.AppendFrameSummary(fs);
    EXPECT_EQ(vm.totalFrames(), 1);
}

TEST(DashboardViewModelTest, OkAndNgCountsTrackCorrectly) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;

    sai::visualization::FrameSummary ok_frame;
    ok_frame.frame_id = 1;
    ok_frame.verdict = "OK";
    ok_frame.timestamp = std::chrono::system_clock::now();

    sai::visualization::FrameSummary ng_frame;
    ng_frame.frame_id = 2;
    ng_frame.verdict = "NG";
    ng_frame.timestamp = std::chrono::system_clock::now();

    vm.AppendFrameSummary(ok_frame);
    vm.AppendFrameSummary(ng_frame);

    EXPECT_EQ(vm.totalFrames(), 2);
    EXPECT_EQ(vm.okFrames(), 1);
    EXPECT_EQ(vm.ngFrames(), 1);
    EXPECT_GT(vm.ppm(), 0.0);
}

TEST(DashboardViewModelTest, PpmIsZeroForAllOkFrames) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;

    for (int i = 0; i < 10; i++) {
        sai::visualization::FrameSummary fs;
        fs.frame_id = i;
        fs.verdict = "OK";
        fs.timestamp = std::chrono::system_clock::now();
        vm.AppendFrameSummary(fs);
    }
    EXPECT_EQ(vm.totalFrames(), 10);
    EXPECT_EQ(vm.ppm(), 0.0);
}

TEST(DashboardViewModelTest, MaxBufferEvictsOldEntries) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;
    for (int i = 0; i < 1100; i++) {
        sai::visualization::FrameSummary fs;
        fs.frame_id = i;
        fs.verdict = "OK";
        fs.timestamp = std::chrono::system_clock::now();
        vm.AppendFrameSummary(fs);
    }
    // Should not crash, should cap at kMaxFrameSummaries (1000)
    EXPECT_EQ(vm.totalFrames(), 1100);
}

TEST(DashboardViewModelTest, DefectTypeModelPopulated) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;

    sai::visualization::FrameSummary fs;
    fs.frame_id = 1;
    fs.verdict = "NG";
    fs.defect_types = {"scratch", "dent", "scratch"};
    fs.timestamp = std::chrono::system_clock::now();

    vm.AppendFrameSummary(fs);
    EXPECT_EQ(vm.totalFrames(), 1);

    auto* model = vm.defectTypeModel();
    EXPECT_NE(model, nullptr);
}

TEST(DashboardViewModelTest, SignalEmittedOnSummary) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::DashboardViewModel vm;

    int emitted = 0;
    QObject::connect(&vm, &sai::visualization::DashboardViewModel::summaryChanged,
                     [&]() { emitted++; });

    sai::visualization::FrameSummary fs;
    fs.frame_id = 1;
    fs.verdict = "WARN";
    fs.timestamp = std::chrono::system_clock::now();
    vm.AppendFrameSummary(fs);

    EXPECT_EQ(emitted, 1);
}
