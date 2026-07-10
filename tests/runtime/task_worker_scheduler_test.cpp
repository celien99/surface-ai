#include <sai/runtime/task.h>
#include <sai/runtime/worker_pool.h>
#include <sai/runtime/task_scheduler.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>

namespace {

using sai::ErrorCode;
using sai::Registry;
using sai::Result;
using sai::TypeId;
using sai::runtime::Task;
using sai::runtime::TaskScheduler;
using sai::runtime::WorkerPool;

// ---------------------------------------------------------------------------
// Task<T> / TaskPromise<T> machinery (brief Step 1)
// ---------------------------------------------------------------------------

Task<int> ReturnsSuccess() { co_return Result<int>(42); }

Task<int> ReturnsError() {
    co_return tl::make_unexpected(sai::ErrorInfo{
        ErrorCode::Core_Unknown,
        "deliberate failure",
        std::source_location::current(),
    });
}

Task<int> ReflectsStopToken(std::stop_token token) {
    // Two checkpoints, mirroring the "before" (construction) and "after"
    // (post request_stop) states the brief asks for.
    co_return Result<int>(token.stop_requested() ? 1 : 0);
}

TEST(TaskTest, CoReturnsSuccessResult) {
    Task<int> handle = ReturnsSuccess();
    handle.resume();

    ASSERT_TRUE(handle.done());
    Result<int> result = handle.promise().GetResult();
    handle.destroy();  // Manual-destroy contract (§11 Memory): result already taken.

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

TEST(TaskTest, CoReturnsErrorResult) {
    Task<int> handle = ReturnsError();
    handle.resume();

    ASSERT_TRUE(handle.done());
    Result<int> result = handle.promise().GetResult();
    handle.destroy();

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Core_Unknown);
}

TEST(TaskTest, GetStopTokenReflectsConstructionAndRequestStop) {
    std::stop_source source;
    Task<int> handle = ReflectsStopToken(source.get_token());

    // Reflects the token passed at construction, before any request_stop().
    EXPECT_FALSE(handle.promise().GetStopToken().stop_requested());

    source.request_stop();

    // Same promise, same underlying stop_state — stop_requested() flips
    // without re-constructing the coroutine.
    EXPECT_TRUE(handle.promise().GetStopToken().stop_requested());

    handle.resume();
    ASSERT_TRUE(handle.done());
    Result<int> result = handle.promise().GetResult();
    handle.destroy();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1);
}

// ---------------------------------------------------------------------------
// WorkerPool (brief Step 3)
// ---------------------------------------------------------------------------

// A trivial resumable "coroutine handle" stand-in: task.h's Task<T> is the
// framework-wide coroutine handle type, but WorkerPool::TryEnqueue's
// signature (§4) takes a bare std::coroutine_handle<> — any coroutine handle
// works, so these tests drive it with Task<void> coroutines that increment a
// shared counter, matching how TaskScheduler/PipelineExecutor will use it in
// later tasks.
//
// These tests never poll this counter (or read a resumed handle's
// done()/promise()) while the WorkerPool that resumed it is still alive.
// Instead they always destroy the WorkerPool first (letting it go out of
// scope) and only inspect state afterwards. This is deliberate, not just
// style: WorkerPool's destructor request_stop()s then join()s every worker
// jthread, and WorkerLoop's wait predicate keeps draining the queue for as
// long as it is non-empty (see worker_pool.cpp) even after a stop has been
// requested, so every handle enqueued before the pool starts destructing is
// guaranteed to have been popped and resume()'d by the time join() returns.
// std::thread::join() (which jthread's destructor performs) is one of the
// few operations the standard guarantees to be a full happens-before edge
// over *everything* the joined thread did — not just one atomic variable.
// An earlier draft of these tests instead had each coroutine release-store a
// per-handle "done" flag and polled that from the main thread while the
// pool kept running; ThreadSanitizer correctly flagged that as a genuine
// data race, because nothing inside a coroutine body can run after the
// compiler-generated code that actually marks the coroutine_handle done() at
// final_suspend — so no signal raised *from inside* the coroutine can ever
// certify that final state to another thread. Relying on join() instead
// sidesteps the question entirely.
Task<void> IncrementCounter(std::atomic<int>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
    co_return Result<void>{};
}

TEST(WorkerPoolTest, ConstructsWithThreadCountAndQueueCapacity) {
    WorkerPool pool(2, 4);

    EXPECT_EQ(pool.ThreadCount(), 2u);
    EXPECT_EQ(pool.PendingCount(), 0u);
}

TEST(WorkerPoolTest, TryEnqueueResumesHandleOnPoolThread) {
    // Queue capacity is deliberately >= kTaskCount below: this test's only
    // concern is "does an enqueued handle actually get resumed by a pool
    // thread", not queue-full behavior (that has its own dedicated test).
    constexpr int kTaskCount = 8;
    std::atomic<int> counter{0};

    std::vector<Task<void>> handles;
    handles.reserve(kTaskCount);
    {
        WorkerPool pool(2, kTaskCount);
        for (int i = 0; i < kTaskCount; ++i) {
            Task<void> handle = IncrementCounter(counter);
            ASSERT_TRUE(pool.TryEnqueue(handle).has_value());
            handles.push_back(handle);
        }
        // pool's destructor runs at the end of this block: it joins both
        // worker threads, which (per WorkerLoop's drain-until-empty
        // behavior) resume()s every handle enqueued above before join()
        // returns — see this section's header comment for why that join is
        // exactly what makes the reads below race-free.
    }

    EXPECT_EQ(counter.load(std::memory_order_relaxed), kTaskCount);
    for (Task<void> handle : handles) {
        ASSERT_TRUE(handle.done());
        ASSERT_TRUE(handle.promise().GetResult().has_value());
        handle.destroy();
    }
}

// Blocks on a manually-releasable gate — used to deliberately keep every
// worker thread busy so a queue-full test is a genuine capacity test, not a
// timing race.
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool open = false;

    void Wait() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return open; });
    }

    void Open() {
        std::unique_lock lock(mutex);
        open = true;
        cv.notify_all();
    }
};

// `started` signals "this coroutine has entered gate.Wait()" so the test
// knows the queue is genuinely isolated from draining before it starts
// filling the queue to capacity. A relaxed fetch_add is enough here (unlike
// the done-state of a completed coroutine — see IncrementCounter's comment
// above), because this test never reads anything about the coroutine's
// *completion* through `started`; it only waits for both worker threads to
// have entered gate.Wait(), then later synchronizes with actual completion
// by joining pool's threads (destroying pool) before touching any handle.
Task<void> BlockOnGate(Gate& gate, std::atomic<int>& started) {
    started.fetch_add(1, std::memory_order_relaxed);
    gate.Wait();
    co_return Result<void>{};
}

TEST(WorkerPoolTest, TryEnqueueReturnsQueueFullWhenCapacityExceeded) {
    constexpr std::size_t kThreadCount = 2;
    constexpr std::size_t kQueueCapacity = 4;

    Gate gate;
    std::atomic<int> started{0};
    std::atomic<int> filler_counter{0};
    std::vector<Task<void>> blocking_handles;
    std::vector<Task<void>> filler_handles;
    std::optional<Result<void>> overflow_result;
    std::optional<Task<void>> overflow_handle;

    {
        WorkerPool pool(kThreadCount, kQueueCapacity);

        // Occupy both worker threads with tasks that block on the gate, so
        // they cannot drain the queue while the test enqueues more work.
        for (std::size_t i = 0; i < kThreadCount; ++i) {
            Task<void> handle = BlockOnGate(gate, started);
            ASSERT_TRUE(pool.TryEnqueue(handle).has_value());
            blocking_handles.push_back(handle);
        }

        // Wait until both worker threads have actually picked up their
        // blocking task (not just been queued) before treating the queue as
        // isolated from draining. This poll only needs to observe "has
        // gate.Wait() been entered" — it does not need to observe the
        // coroutine's completion (that comes later, via join), so a relaxed
        // spin on `started` is sufficient here.
        const auto started_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (started.load(std::memory_order_relaxed) < static_cast<int>(kThreadCount) &&
               std::chrono::steady_clock::now() < started_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ASSERT_EQ(started.load(std::memory_order_relaxed), static_cast<int>(kThreadCount));

        // Now fill the queue to capacity (both threads gated, so these
        // never get to run while pool is still alive).
        for (std::size_t i = 0; i < kQueueCapacity; ++i) {
            Task<void> handle = IncrementCounter(filler_counter);
            ASSERT_TRUE(pool.TryEnqueue(handle).has_value());
            filler_handles.push_back(handle);
        }

        EXPECT_EQ(pool.PendingCount(), kQueueCapacity);

        // The queue is now genuinely full (both threads gated, capacity
        // used up) — the next submission must be rejected.
        overflow_handle = IncrementCounter(filler_counter);
        overflow_result = pool.TryEnqueue(*overflow_handle);

        // Release the gate so the pool can drain before it destructs below
        // (WorkerPool's destructor joins its threads unconditionally; if
        // they were still gated, that join would hang forever).
        gate.Open();

        // pool's destructor runs at the end of this block: join()ing both
        // worker threads guarantees every blocking_handle and filler_handle
        // enqueued above has actually been resume()'d and reached
        // final_suspend by the time the block exits — see
        // IncrementCounter's comment for why that join is what makes
        // reading these handles afterwards race-free.
    }

    ASSERT_TRUE(overflow_result.has_value());  // Submitted inside the block above.
    ASSERT_FALSE(overflow_result->has_value());
    EXPECT_EQ(overflow_result->error().code, ErrorCode::Runtime_QueueFull);
    EXPECT_EQ(filler_counter.load(std::memory_order_relaxed), static_cast<int>(kQueueCapacity));

    for (Task<void> handle : blocking_handles) {
        ASSERT_TRUE(handle.done());
        ASSERT_TRUE(handle.promise().GetResult().has_value());
        handle.destroy();
    }
    for (Task<void> handle : filler_handles) {
        ASSERT_TRUE(handle.done());
        ASSERT_TRUE(handle.promise().GetResult().has_value());
        handle.destroy();
    }
    // The rejected overflow handle was never enqueued and never resumed —
    // it is still a valid, suspended coroutine frame owned solely by this
    // test, so it must be destroyed here too (never submitted, never run).
    ASSERT_TRUE(overflow_handle.has_value());
    overflow_handle->destroy();
}

// ---------------------------------------------------------------------------
// TaskScheduler (brief Step 4)
// ---------------------------------------------------------------------------

TEST(TaskSchedulerTest, SubmitReachesRegisteredStageWorkerPool) {
    const TypeId stage_id = sai::detail::Fnv1aHash("Capture");
    std::atomic<int> counter{0};
    Task<void> handle = IncrementCounter(counter);
    std::optional<Result<void>> submit_result;

    {
        // Registry<WorkerPool> holds the WorkerPool via shared_ptr, and
        // TaskScheduler only holds a reference to the registry, so both are
        // scoped here: dropping the registry drops the pool's last
        // shared_ptr, running WorkerPool's destructor (join) before this
        // block exits — see IncrementCounter's comment for why that join,
        // not a polled flag, is what makes reading `handle` below
        // race-free.
        Registry<WorkerPool> registry;
        ASSERT_TRUE(registry.Register(stage_id, std::make_shared<WorkerPool>(1, 4)).has_value());
        TaskScheduler scheduler(registry);

        submit_result = scheduler.Submit(stage_id, handle);
    }

    ASSERT_TRUE(submit_result.has_value());
    ASSERT_TRUE(submit_result->has_value());
    EXPECT_EQ(counter.load(std::memory_order_relaxed), 1);
    ASSERT_TRUE(handle.done());
    ASSERT_TRUE(handle.promise().GetResult().has_value());
    handle.destroy();
}

TEST(TaskSchedulerTest, SubmitToUnregisteredStagePropagatesRegistryResolveError) {
    Registry<WorkerPool> registry;  // Nothing registered.
    TaskScheduler scheduler(registry);

    const TypeId unregistered_stage_id = sai::detail::Fnv1aHash("Inference");
    std::atomic<int> counter{0};
    Task<void> handle = IncrementCounter(counter);

    Result<void> submit_result = scheduler.Submit(unregistered_stage_id, handle);

    ASSERT_FALSE(submit_result.has_value());
    EXPECT_EQ(submit_result.error().code, ErrorCode::Core_TypeNotFound);

    // Never submitted (Registry::Resolve failed before reaching any
    // WorkerPool), so this handle was never resumed — destroy it directly.
    handle.destroy();
}

}  // namespace
