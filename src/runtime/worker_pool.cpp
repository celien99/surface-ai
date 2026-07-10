#include <sai/runtime/worker_pool.h>

#include <source_location>
#include <utility>

namespace sai::runtime {

WorkerPool::WorkerPool(std::size_t thread_count, std::size_t queue_capacity) noexcept
    : thread_count_(thread_count), queue_capacity_(queue_capacity) {
    threads_.reserve(thread_count_);
    for (std::size_t i = 0; i < thread_count_; ++i) {
        threads_.emplace_back([this](std::stop_token stop_token) { WorkerLoop(stop_token); });
    }
}

// jthread's destructor calls request_stop() then join() for each thread in
// turn; WorkerLoop's condition_variable_any::wait(lock, stop_token, pred)
// wakes immediately when its own thread's stop is requested (see header
// comment), so no explicit notify is needed here.
WorkerPool::~WorkerPool() noexcept = default;

auto WorkerPool::TryEnqueue(std::coroutine_handle<> handle) noexcept -> Result<void> {
    std::unique_lock lock(queue_mutex_);
    if (queue_.size() >= queue_capacity_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_QueueFull,
            "worker pool queue is at capacity",
            std::source_location::current(),
        });
    }
    queue_.push_back(handle);
    lock.unlock();
    queue_condition_.notify_one();
    return {};
}

auto WorkerPool::ThreadCount() const noexcept -> std::size_t { return thread_count_; }

auto WorkerPool::PendingCount() const noexcept -> std::size_t {
    std::unique_lock lock(queue_mutex_);
    return queue_.size();
}

void WorkerPool::WorkerLoop(std::stop_token stop_token) {
    while (true) {
        std::coroutine_handle<> handle;
        {
            std::unique_lock lock(queue_mutex_);
            const bool has_work =
                queue_condition_.wait(lock, stop_token, [this] { return !queue_.empty(); });
            if (!has_work) {
                return;  // Stop requested and no work left in the queue.
            }
            handle = queue_.front();
            queue_.pop_front();
        }
        handle.resume();
    }
}

}  // namespace sai::runtime
