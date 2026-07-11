#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace sai::device {

template <typename T>
class RingBuffer final {
public:
    explicit RingBuffer(std::size_t capacity) noexcept
        : slots_(capacity)
        , capacity_(capacity)
    {}

    auto Push(T item) noexcept -> void {
        if (capacity_ == 0) {
            ++dropped_count_;
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == capacity_) {
            // Overwrite oldest unread element.
            slots_[tail_].reset();
            tail_ = (tail_ + 1) % capacity_;
            ++dropped_count_;
        }
        slots_[head_] = std::move(item);
        head_ = (head_ + 1) % capacity_;
        if (count_ < capacity_) {
            ++count_;
        }
    }

    [[nodiscard]] auto TryPop() noexcept -> std::optional<T> {
        if (capacity_ == 0) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) {
            return std::nullopt;
        }
        T result = std::move(*slots_[tail_]);
        slots_[tail_].reset();
        tail_ = (tail_ + 1) % capacity_;
        --count_;
        return result;
    }

    [[nodiscard]] auto Capacity() const noexcept -> std::size_t { return capacity_; }

    [[nodiscard]] auto Size() const noexcept -> std::size_t {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

    [[nodiscard]] auto DroppedCount() const noexcept -> std::size_t { return dropped_count_; }

    RingBuffer(const RingBuffer&) = delete;
    auto operator=(const RingBuffer&) -> RingBuffer& = delete;

private:
    std::vector<std::optional<T>> slots_;
    std::size_t capacity_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    std::size_t dropped_count_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace sai::device
