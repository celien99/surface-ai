#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>

#include <sai/core/error.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai::pipeline {

namespace detail {

// Cache line padding to prevent false sharing between head and tail.
// arm64 cache line = 128 bytes; x86-64 = 64 bytes. Use 128 to cover both.
static constexpr size_t kCacheLineSize = 128;

// Bounded SPSC ring buffer with backpressure. Single-producer-single-consumer
// by design — the caller is responsible for ensuring only one thread calls
// TryPush / PushBlocking and only one calls TryPop / PopBlocking.
// For MPSC (fan-in), wrap with a mutex on the producer side.
//
// Internally allocates capacity + 1 slots; one slot is always kept empty to
// distinguish full from empty, so the effective capacity is `capacity`.
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity, BackpressurePolicy policy)
        : logical_capacity_(capacity)
        , policy_(policy)
        , buffer_(std::make_unique<std::unique_ptr<T>[]>(capacity + 1))
        , head_(0)
        , tail_(0) {}

    auto TryPush(std::unique_ptr<T> item) noexcept -> bool {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % (logical_capacity_ + 1);

        if (next_tail == head_.load(std::memory_order_acquire)) {
            // Full
            if (policy_ == BackpressurePolicy::DropOldest) {
                // Discard the oldest: advance head by one
                size_t current_head = head_.load(std::memory_order_relaxed);
                head_.store((current_head + 1) % (logical_capacity_ + 1),
                            std::memory_order_release);
                // Now we have space — write at tail
                buffer_[current_tail] = std::move(item);
                tail_.store(next_tail, std::memory_order_release);
                return true;
            }
            return false;
        }

        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    auto TryPop() noexcept -> std::unique_ptr<T> {
        size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return nullptr;  // Empty
        }

        auto item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) % (logical_capacity_ + 1),
                    std::memory_order_release);
        return item;
    }

    auto Capacity() const noexcept -> size_t { return logical_capacity_; }

    auto Depth() const noexcept -> size_t {
        size_t t = tail_.load(std::memory_order_acquire);
        size_t h = head_.load(std::memory_order_acquire);
        if (t >= h) return t - h;
        return (logical_capacity_ + 1) - h + t;
    }

    // For PushBlocking / PopBlocking (external CV)
    auto IsFull() const noexcept -> bool {
        size_t next = (tail_.load(std::memory_order_relaxed) + 1)
                      % (logical_capacity_ + 1);
        return next == head_.load(std::memory_order_relaxed);
    }

    auto IsEmpty() const noexcept -> bool {
        return head_.load(std::memory_order_relaxed)
               == tail_.load(std::memory_order_relaxed);
    }

private:
    size_t logical_capacity_;
    BackpressurePolicy policy_;
    std::unique_ptr<std::unique_ptr<T>[]> buffer_;

    // Padded to separate cache lines to avoid false sharing
    alignas(kCacheLineSize) std::atomic<size_t> head_;
    alignas(kCacheLineSize) std::atomic<size_t> tail_;
};

}  // namespace detail

template <typename T>
class StageQueue {
public:
    static auto Create(size_t capacity, BackpressurePolicy policy)
        -> Result<std::unique_ptr<StageQueue>> {
        if (capacity == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Scheduler_QueueCreateFailed,
                "Queue capacity must be > 0"
            });
        }
        return std::unique_ptr<StageQueue>(
            new StageQueue(capacity, policy));
    }

    auto TryPush(std::unique_ptr<T> item) -> bool {
        bool ok = ring_.TryPush(std::move(item));
        if (ok) {
            cv_.notify_one();  // wake pop side (PopBlocking or PushBlocking)
        }
        return ok;
    }

    auto PushBlocking(std::unique_ptr<T> item) -> void {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !ring_.IsFull(); });
        ring_.TryPush(std::move(item));  // guaranteed to succeed
        cv_.notify_one();  // wake pop side
    }

    auto TryPop() -> std::unique_ptr<T> {
        auto item = ring_.TryPop();
        if (item) {
            cv_.notify_one();  // wake push side (PushBlocking or PopBlocking)
        }
        return item;
    }

    auto PopBlocking() -> std::unique_ptr<T> {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !ring_.IsEmpty(); });
        auto item = ring_.TryPop();  // guaranteed to succeed
        cv_.notify_one();  // wake push side
        return item;
    }

    auto Depth() const noexcept -> size_t { return ring_.Depth(); }
    auto Capacity() const noexcept -> size_t { return ring_.Capacity(); }

private:
    StageQueue(size_t capacity, BackpressurePolicy policy)
        : ring_(capacity, policy) {}

    detail::RingBuffer<T> ring_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace sai::pipeline
