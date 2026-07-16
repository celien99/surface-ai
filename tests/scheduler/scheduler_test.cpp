#include <gtest/gtest.h>

#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/pipeline_config.h>

#include "src/scheduler/scheduler.h"

namespace sai::pipeline {
namespace {

TEST(SchedulerTest, AllocateAndResolve) {
    Scheduler scheduler;

    BackpressureConfig bp;
    bp.default_policy = BackpressurePolicy::Block;

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "inference";
    s.type = StageType::Inference;
    stages.push_back(s);

    auto alloc_result = scheduler.Allocate(stages, bp);
    ASSERT_TRUE(alloc_result.has_value());

    auto pool_result = scheduler.PoolFor(StageType::Inference);
    ASSERT_TRUE(pool_result.has_value());
    EXPECT_GT((*pool_result)->ThreadCount(), 0);
}

TEST(SchedulerTest, PoolForWithoutAllocateReturnsError) {
    Scheduler scheduler;

    auto result = scheduler.PoolFor(StageType::Detect);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, DeallocateClearsMapping) {
    Scheduler scheduler;

    BackpressureConfig bp;
    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "capture";
    s.type = StageType::Capture;
    stages.push_back(s);

    ASSERT_TRUE(scheduler.Allocate(stages, bp).has_value());
    ASSERT_TRUE(scheduler.PoolFor(StageType::Capture).has_value());

    scheduler.Deallocate();
    auto result = scheduler.PoolFor(StageType::Capture);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, CaptureAndPreprocessSharePool) {
    Scheduler scheduler;

    BackpressureConfig bp;
    std::vector<StageConfig> stages;
    {
        StageConfig s;
        s.id = "capture";
        s.type = StageType::Capture;
        stages.push_back(s);
    }
    {
        StageConfig s;
        s.id = "preprocess";
        s.type = StageType::Preprocess;
        stages.push_back(s);
    }

    ASSERT_TRUE(scheduler.Allocate(stages, bp).has_value());

    // Both should resolve to the same pool (Capture + Preprocess share "Capture" pool)
    auto p1 = scheduler.PoolFor(StageType::Capture);
    auto p2 = scheduler.PoolFor(StageType::Preprocess);
    ASSERT_TRUE(p1.has_value());
    ASSERT_TRUE(p2.has_value());
    EXPECT_EQ(*p1, *p2);  // same pool instance
}

TEST(SchedulerTest, StageTypeToPoolKeyMapping) {
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Capture), "Capture");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Preprocess), "Capture");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Inference), "Inference");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Detect), "Inference");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::RuleEval), "Reason");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Reason), "Reason");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Export), "IO");
    EXPECT_EQ(Scheduler::StageTypeToPoolKey(StageType::Custom), "Background");
}

TEST(SchedulerTest, PoolConfigForKeyDefaults) {
    auto cap_cfg = Scheduler::PoolConfigForKey("Capture");
    EXPECT_EQ(cap_cfg.threads, 2);
    EXPECT_EQ(cap_cfg.queue_capacity, 8);

    auto inf_cfg = Scheduler::PoolConfigForKey("Inference");
    EXPECT_EQ(inf_cfg.threads, 1);
    EXPECT_EQ(inf_cfg.queue_capacity, 4);

    auto io_cfg = Scheduler::PoolConfigForKey("IO");
    EXPECT_EQ(io_cfg.threads, 1);
    EXPECT_EQ(io_cfg.queue_capacity, 32);

    auto unknown_cfg = Scheduler::PoolConfigForKey("NonExistent");
    EXPECT_EQ(unknown_cfg.threads, 1);
    EXPECT_EQ(unknown_cfg.queue_capacity, 8);
}

}  // namespace
}  // namespace sai::pipeline
