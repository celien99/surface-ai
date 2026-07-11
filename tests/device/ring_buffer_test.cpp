#include <sai/device/ring_buffer.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct TestFrame {
    int id;
    std::unique_ptr<int> tag;

    TestFrame(int i, int t) : id(i), tag(std::make_unique<int>(t)) {}

    // Move-only: no copy.
    TestFrame(const TestFrame&) = delete;
    TestFrame& operator=(const TestFrame&) = delete;
    TestFrame(TestFrame&&) noexcept = default;
    TestFrame& operator=(TestFrame&&) noexcept = default;
};

}  // namespace

// ============================================================================
// Fresh buffer is empty
// ============================================================================

TEST(RingBuffer, FreshBufferIsEmpty) {
    sai::device::RingBuffer<int> rb(4);
    EXPECT_EQ(rb.Size(), 0U);
    EXPECT_EQ(rb.TryPop(), std::nullopt);
    EXPECT_EQ(rb.Capacity(), 4U);
    EXPECT_EQ(rb.DroppedCount(), 0U);
}

// ============================================================================
// Push then pop maintains FIFO order
// ============================================================================

TEST(RingBuffer, PushPopFifo) {
    sai::device::RingBuffer<int> rb(4);
    rb.Push(10);
    rb.Push(20);
    rb.Push(30);
    EXPECT_EQ(rb.Size(), 3U);
    EXPECT_EQ(rb.TryPop(), 10);
    EXPECT_EQ(rb.TryPop(), 20);
    EXPECT_EQ(rb.TryPop(), 30);
    EXPECT_EQ(rb.TryPop(), std::nullopt);
}

// ============================================================================
// Overwrites oldest when full (verbatim from brief)
// ============================================================================

TEST(RingBuffer, OverwritesOldestWhenFull) {
    sai::device::RingBuffer<int> rb(2);
    rb.Push(1); rb.Push(2); rb.Push(3);           // 1 overwritten
    EXPECT_EQ(rb.DroppedCount(), 1U);
    EXPECT_EQ(rb.Size(), 2U);
    EXPECT_EQ(rb.TryPop(), 2);
    EXPECT_EQ(rb.TryPop(), 3);
    EXPECT_EQ(rb.TryPop(), std::nullopt);
}

// ============================================================================
// Size never exceeds Capacity
// ============================================================================

TEST(RingBuffer, SizeNeverExceedsCapacity) {
    sai::device::RingBuffer<int> rb(3);
    for (int i = 0; i < 100; ++i) {
        rb.Push(i);
        EXPECT_LE(rb.Size(), rb.Capacity());
    }
}

// ============================================================================
// DroppedCount increments per overwrite
// ============================================================================

TEST(RingBuffer, DroppedCountIncrementsPerOverwrite) {
    sai::device::RingBuffer<int> rb(2);
    rb.Push(1);
    rb.Push(2);
    EXPECT_EQ(rb.DroppedCount(), 0U);
    rb.Push(3);  // overwrites 1
    EXPECT_EQ(rb.DroppedCount(), 1U);
    rb.Push(4);  // overwrites 2
    EXPECT_EQ(rb.DroppedCount(), 2U);
    rb.Push(5);  // overwrites 3
    EXPECT_EQ(rb.DroppedCount(), 3U);
}

// ============================================================================
// Move-only types supported
// ============================================================================

TEST(RingBuffer, MoveOnlyTypes) {
    sai::device::RingBuffer<TestFrame> rb(2);
    rb.Push(TestFrame{1, 100});
    rb.Push(TestFrame{2, 200});

    auto f1 = rb.TryPop();
    ASSERT_TRUE(f1.has_value());
    EXPECT_EQ(f1->id, 1);
    EXPECT_NE(f1->tag, nullptr);
    EXPECT_EQ(*f1->tag, 100);

    auto f2 = rb.TryPop();
    ASSERT_TRUE(f2.has_value());
    EXPECT_EQ(f2->id, 2);
    EXPECT_NE(f2->tag, nullptr);
    EXPECT_EQ(*f2->tag, 200);
}

// ============================================================================
// Move-only overwrite drops oldest
// ============================================================================

TEST(RingBuffer, MoveOnlyOverwritesOldest) {
    sai::device::RingBuffer<TestFrame> rb(2);
    rb.Push(TestFrame{1, 10});
    rb.Push(TestFrame{2, 20});
    rb.Push(TestFrame{3, 30});  // overwrites frame 1

    EXPECT_EQ(rb.DroppedCount(), 1U);
    auto f = rb.TryPop();
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->id, 2);
    EXPECT_EQ(*f->tag, 20);
}

// ============================================================================
// Concurrent single-producer / multi-consumer
// ============================================================================

TEST(RingBuffer, ConcurrentProducerConsumer) {
    static constexpr int kTotalItems = 10000;
    static constexpr int kNumConsumers = 3;

    sai::device::RingBuffer<int> rb(kTotalItems);

    std::atomic<int> consumed_count{0};
    std::atomic<int> sum{0};
    std::atomic<bool> done{false};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 1; i <= kTotalItems; ++i) {
            rb.Push(i);
        }
        done = true;
    });

    // Consumer threads
    std::vector<std::thread> consumers;
    for (int c = 0; c < kNumConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (!done || rb.Size() > 0) {
                auto item = rb.TryPop();
                if (item.has_value()) {
                    consumed_count.fetch_add(1, std::memory_order_relaxed);
                    sum.fetch_add(*item, std::memory_order_relaxed);
                }
            }
        });
    }

    producer.join();
    for (auto& t : consumers) {
        t.join();
    }

    // All items accounted for exactly once.
    EXPECT_EQ(consumed_count.load(), kTotalItems);
    // Sum of 1..N = N*(N+1)/2
    int expected_sum = kTotalItems * (kTotalItems + 1) / 2;
    EXPECT_EQ(sum.load(), expected_sum);
}

// ============================================================================
// Capacity of 0 — trivial no-op
// ============================================================================

TEST(RingBuffer, ZeroCapacity) {
    sai::device::RingBuffer<int> rb(0);
    EXPECT_EQ(rb.Capacity(), 0U);
    EXPECT_EQ(rb.Size(), 0U);
    rb.Push(42);  // should be a no-op
    EXPECT_EQ(rb.Size(), 0U);
    EXPECT_EQ(rb.TryPop(), std::nullopt);
    EXPECT_EQ(rb.DroppedCount(), 1U);  // Push drops when capacity==0
}
