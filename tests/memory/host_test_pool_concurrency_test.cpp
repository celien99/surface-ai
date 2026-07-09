#include "host_test_pool.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <sai/memory/memory_pool.h>

namespace {

using sai::memory::MemoryPoolConfig;
using sai::test::HostTestPool;

constexpr int kThreadCount = 8;
constexpr int kIterationsPerThread = 20000;
constexpr std::size_t kSlabCount = 4;
constexpr std::size_t kSlabSize = sizeof(std::uint64_t);

}  // namespace

// Regression guard for the free-list ABA fix (1.5-memory.md §9): several
// threads hammer Acquire/Release concurrently against one shared pool. Each
// successful Acquire stamps its slab with a globally unique tag, yields
// (widening the window a still-buggy bare-pointer free list would need to
// hand the same slab to two live handles at once), then reads the tag back.
// If any other thread's handle pointed at the same physical slab in the
// meantime, the read-back won't match what was just written.
//
// With the tagged-head fix in place this is expected to pass reliably every
// run — it isn't a proof the old bare-pointer code would fail here, just the
// going-forward concurrency coverage the task review asked for.
TEST(HostTestPoolConcurrencyTest, ParallelAcquireReleaseNeverCorruptsOrDoublesUpSlabs) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = kSlabSize, .slab_count = kSlabCount});
    std::atomic<std::uint64_t> next_tag{1};
    std::atomic<bool> corrupted{false};

    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int t = 0; t < kThreadCount; ++t) {
        workers.emplace_back([&pool, &next_tag, &corrupted] {
            for (int i = 0; i < kIterationsPerThread; ++i) {
                auto handle = pool.Acquire(kSlabSize);
                if (!handle.has_value()) {
                    continue;  // Pool momentarily exhausted under contention; not an error.
                }

                const std::uint64_t tag = next_tag.fetch_add(1, std::memory_order_relaxed);
                std::memcpy(handle->Get(), &tag, sizeof(tag));
                std::this_thread::yield();

                std::uint64_t observed = 0;
                std::memcpy(&observed, handle->Get(), sizeof(observed));
                if (observed != tag) {
                    corrupted.store(true, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_FALSE(corrupted.load(std::memory_order_relaxed));
    EXPECT_EQ(pool.AvailableSlabCount(), kSlabCount);
}
