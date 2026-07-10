#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/worker_pool.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------

#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <sai/core/error.h>

namespace sai::runtime {

// Fixed-thread-count worker pool; one instance per Pipeline stage
// (Capture/Inference/Retrieval/Reason/IO). thread_count is fixed at
// construction (from YAML config), never adjusted at runtime (§3 Design).
class WorkerPool final {
public:
    // thread_count: number of worker threads this pool holds for its whole
    // lifetime. queue_capacity: bound on the internal queue; TryEnqueue
    // rejects once the queue is at this capacity (backpressure is the
    // caller's — typically TaskScheduler's — decision, this class only
    // reports "can another item fit").
    WorkerPool(std::size_t thread_count, std::size_t queue_capacity) noexcept;
    ~WorkerPool() noexcept;

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;

    // Enqueues a resumable coroutine handle; some idle worker thread later
    // pops it and calls resume(). Returns Runtime_QueueFull without
    // blocking the caller if the queue is already at capacity.
    [[nodiscard]] auto TryEnqueue(std::coroutine_handle<> handle) noexcept -> Result<void>;

    [[nodiscard]] auto ThreadCount() const noexcept -> std::size_t;
    [[nodiscard]] auto PendingCount() const noexcept -> std::size_t;

private:
    void WorkerLoop(std::stop_token stop_token);

    std::size_t thread_count_;
    std::size_t queue_capacity_;

    mutable std::mutex queue_mutex_;
    std::condition_variable_any queue_condition_;
    std::deque<std::coroutine_handle<>> queue_;

    // Declared last so it is destroyed (joined) first: members are torn down
    // in reverse declaration order, and WorkerLoop (running on these
    // threads) reads queue_mutex_/queue_condition_/queue_ until it observes
    // its own thread's stop_token, so those primitives must outlive every
    // jthread's join() — not the other way around.
    std::vector<std::jthread> threads_;  // jthread's built-in stop_token drives overall shutdown.
};

}  // namespace sai::runtime
