#include <gtest/gtest.h>

#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/pipeline_config.h>

#include "src/scheduler/scheduler.h"

namespace sai::pipeline {
namespace {

TEST(SchedulerTest, AllocateAndResolve) {
    // Set up a Registry<WorkerPool> with required pools
    Registry<runtime::WorkerPool> pools;
    auto pool = std::make_shared<runtime::WorkerPool>(2, 8);
    ASSERT_TRUE(pools.Register(sai::detail::Fnv1aHash("Inference"), pool).has_value());

    BackpressureConfig bp;
    bp.default_policy = BackpressurePolicy::Block;

    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "inference";
    s.type = StageType::Inference;
    stages.push_back(s);

    auto alloc_result = scheduler.Allocate(stages);
    ASSERT_TRUE(alloc_result.has_value());

    auto pool_result = scheduler.PoolFor(StageType::Inference);
    ASSERT_TRUE(pool_result.has_value());
    EXPECT_EQ((*pool_result)->ThreadCount(), 2);
}

TEST(SchedulerTest, MissingPoolReturnsError) {
    Registry<runtime::WorkerPool> pools;
    // No pools registered
    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "detect";
    s.type = StageType::Detect;
    stages.push_back(s);

    auto result = scheduler.Allocate(stages);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, DeallocateClearsMapping) {
    Registry<runtime::WorkerPool> pools;
    auto pool = std::make_shared<runtime::WorkerPool>(1, 4);
    ASSERT_TRUE(pools.Register(sai::detail::Fnv1aHash("Capture"), pool).has_value());

    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "capture";
    s.type = StageType::Capture;
    stages.push_back(s);

    ASSERT_TRUE(scheduler.Allocate(stages).has_value());
    ASSERT_TRUE(scheduler.PoolFor(StageType::Capture).has_value());

    ASSERT_TRUE(scheduler.Deallocate().has_value());
    auto result = scheduler.PoolFor(StageType::Capture);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, CaptureAndPreprocessSharePool) {
    Registry<runtime::WorkerPool> pools;
    auto capture_pool = std::make_shared<runtime::WorkerPool>(2, 8);
    ASSERT_TRUE(pools.Register(sai::detail::Fnv1aHash("Capture"), capture_pool).has_value());

    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

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

    ASSERT_TRUE(scheduler.Allocate(stages).has_value());

    // Both should resolve to the same pool
    auto p1 = scheduler.PoolFor(StageType::Capture);
    auto p2 = scheduler.PoolFor(StageType::Preprocess);
    ASSERT_TRUE(p1.has_value());
    ASSERT_TRUE(p2.has_value());
    EXPECT_EQ(*p1, *p2);  // same pool instance
}

}  // namespace
}  // namespace sai::pipeline
