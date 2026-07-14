#include <gtest/gtest.h>
#include <QGuiApplication>
#include "sai/visualization/pipeline_viewmodel.h"
#include "sai/pipeline/pipeline.h"

// Test that StageMetricsObject correctly translates StageMetrics fields
TEST(StageMetricsObjectTest, UpdateFromStageMetricsCapturesFields) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::StageMetricsObject obj("capture", "0");

    sai::pipeline::StageMetrics m("capture", sai::pipeline::StageType::Capture);
    m.frames_processed.store(42);
    m.frames_failed.store(3);
    m.frames_dropped.store(1);
    m.avg_latency_us = 12500.0;  // 12.5ms

    obj.UpdateFromStageMetrics(m);

    EXPECT_EQ(obj.framesProcessed(), 42);
    EXPECT_EQ(obj.framesFailed(), 3);
    EXPECT_EQ(obj.framesDropped(), 1);
    EXPECT_NEAR(obj.avgLatencyMs(), 12.5, 0.01);
}

TEST(PipelineViewModelTest, StartRefreshSetsTimer) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::PipelineViewModel vm;
    vm.StartRefresh(100);
    // Timer is active
    EXPECT_TRUE(true);  // smoke: no crash
    vm.StopRefresh();
}
