#include <gtest/gtest.h>
#include <sai/pipeline/stage_queue.h>
#include <thread>
#include <atomic>

namespace sai::pipeline {
namespace {

using IntQueue = StageQueue<int>;

TEST(StageQueueTest, CreateAndCapacity) {
    auto q = IntQueue::Create(8, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());
    EXPECT_EQ((*q)->Capacity(), 8);
    EXPECT_EQ((*q)->Depth(), 0);
}

TEST(StageQueueTest, CreateZeroCapacityFails) {
    auto q = IntQueue::Create(0, BackpressurePolicy::Block);
    EXPECT_FALSE(q.has_value());
    EXPECT_EQ(q.error().code, ErrorCode::Scheduler_QueueCreateFailed);
}

TEST(StageQueueTest, PushPopSingleElement) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    bool pushed = (*q)->TryPush(std::make_unique<int>(42));
    EXPECT_TRUE(pushed);
    EXPECT_EQ((*q)->Depth(), 1);

    auto val = (*q)->TryPop();
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(*val, 42);
    EXPECT_EQ((*q)->Depth(), 0);
}

TEST(StageQueueTest, TryPushFullQueueReturnsFalse) {
    auto q = IntQueue::Create(2, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));
    EXPECT_FALSE((*q)->TryPush(std::make_unique<int>(3)));  // full
    EXPECT_EQ((*q)->Depth(), 2);
}

TEST(StageQueueTest, TryPopEmptyQueueReturnsNull) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    auto val = (*q)->TryPop();
    EXPECT_EQ(val, nullptr);
}

TEST(StageQueueTest, FifoOrdering) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    for (int i = 1; i <= 4; ++i) {
        EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(i)));
    }
    for (int i = 1; i <= 4; ++i) {
        auto val = (*q)->TryPop();
        ASSERT_NE(val, nullptr);
        EXPECT_EQ(*val, i);
    }
}

TEST(StageQueueTest, DropOldestBackpressure) {
    auto q = IntQueue::Create(2, BackpressurePolicy::DropOldest);
    ASSERT_TRUE(q.has_value());

    // Fill queue: [1, 2]
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));
    EXPECT_EQ((*q)->Depth(), 2);

    // Push 3 when full: drop oldest (1), push 3 -> [2, 3]
    bool pushed = (*q)->TryPush(std::make_unique<int>(3));
    EXPECT_TRUE(pushed);
    EXPECT_EQ((*q)->Depth(), 2);

    auto first = (*q)->TryPop();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(*first, 2);  // 1 was dropped

    auto second = (*q)->TryPop();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(*second, 3);
}

TEST(StageQueueTest, SingleProducerSingleConsumer) {
    constexpr size_t kIters = 10'000;
    auto q = IntQueue::Create(64, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> sum_consumed{0};
    size_t sum_produced = 0;

    std::thread producer([&]() {
        for (size_t i = 0; i < kIters; ++i) {
            sum_produced += i;
            while (!(*q)->TryPush(std::make_unique<int>(static_cast<int>(i)))) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        for (size_t i = 0; i < kIters; ++i) {
            std::unique_ptr<int> val;
            while (!(val = (*q)->TryPop())) {
                std::this_thread::yield();
            }
            sum_consumed.fetch_add(*val, std::memory_order_relaxed);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(producer_done.load());
    EXPECT_EQ(sum_produced, sum_consumed.load());
    EXPECT_EQ((*q)->Depth(), 0);
}

}  // namespace
}  // namespace sai::pipeline
