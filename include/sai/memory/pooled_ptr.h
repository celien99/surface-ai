#pragma once

#include <atomic>
#include <cstddef>
#include <utility>

namespace sai::memory {

class IMemoryPool;

// Reference-counted handle: when the count drops to zero, the destructor
// hands the underlying slab back to its source pool instead of deleting the
// memory. The only legal way to construct a non-empty PooledPtr<T> is
// IMemoryPool::Acquire (via the protected factory it exposes to concrete
// pool implementations) — there is no public path that builds one from a
// raw pointer.
template <typename T>
class PooledPtr final {
public:
    PooledPtr() noexcept = default;  // Empty handle: Get() is nullptr, destructor is a no-op.

    PooledPtr(const PooledPtr& other) noexcept;
    auto operator=(const PooledPtr& other) noexcept -> PooledPtr&;
    PooledPtr(PooledPtr&& other) noexcept;
    auto operator=(PooledPtr&& other) noexcept -> PooledPtr&;
    ~PooledPtr() noexcept;

    [[nodiscard]] auto Get() const noexcept -> T* { return data_; }
    [[nodiscard]] auto SizeBytes() const noexcept -> std::size_t { return size_bytes_; }
    [[nodiscard]] auto UseCount() const noexcept -> int {
        return ref_count_ != nullptr ? ref_count_->load(std::memory_order_acquire) : 0;
    }
    [[nodiscard]] auto IsValid() const noexcept -> bool { return data_ != nullptr; }

    auto operator*() const noexcept -> T& { return *data_; }
    auto operator->() const noexcept -> T* { return data_; }

private:
    // Only IMemoryPool's implementations may reach this constructor.
    // owner_pool_ is the callback target on return, not an object whose
    // lifetime PooledPtr manages itself (the pool outlives every handle it
    // hands out, see 1.5-memory.md §11).
    friend class IMemoryPool;
    PooledPtr(T* data, std::size_t size_bytes, IMemoryPool* owner_pool,
              std::atomic<int>* ref_count) noexcept
        : data_(data), size_bytes_(size_bytes), owner_pool_(owner_pool), ref_count_(ref_count) {}

    // Decrements the refcount if this handle is live and, on reaching zero,
    // hands the slab back to owner_pool_. Body is defined out-of-line in
    // memory_pool.h, where IMemoryPool is a complete type.
    void ReleaseIfLive() noexcept;

    T* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    IMemoryPool* owner_pool_ = nullptr;
    std::atomic<int>* ref_count_ = nullptr;  // Points at the count slot pre-allocated with the slab.
};

template <typename T>
PooledPtr<T>::PooledPtr(const PooledPtr& other) noexcept
    : data_(other.data_),
      size_bytes_(other.size_bytes_),
      owner_pool_(other.owner_pool_),
      ref_count_(other.ref_count_) {
    if (ref_count_ != nullptr) {
        ref_count_->fetch_add(1, std::memory_order_acq_rel);
    }
}

template <typename T>
auto PooledPtr<T>::operator=(const PooledPtr& other) noexcept -> PooledPtr& {
    if (this == &other) {
        return *this;
    }
    ReleaseIfLive();
    data_ = other.data_;
    size_bytes_ = other.size_bytes_;
    owner_pool_ = other.owner_pool_;
    ref_count_ = other.ref_count_;
    if (ref_count_ != nullptr) {
        ref_count_->fetch_add(1, std::memory_order_acq_rel);
    }
    return *this;
}

template <typename T>
PooledPtr<T>::PooledPtr(PooledPtr&& other) noexcept
    : data_(other.data_),
      size_bytes_(other.size_bytes_),
      owner_pool_(other.owner_pool_),
      ref_count_(other.ref_count_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.owner_pool_ = nullptr;
    other.ref_count_ = nullptr;
}

template <typename T>
auto PooledPtr<T>::operator=(PooledPtr&& other) noexcept -> PooledPtr& {
    if (this == &other) {
        return *this;
    }
    ReleaseIfLive();
    data_ = other.data_;
    size_bytes_ = other.size_bytes_;
    owner_pool_ = other.owner_pool_;
    ref_count_ = other.ref_count_;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.owner_pool_ = nullptr;
    other.ref_count_ = nullptr;
    return *this;
}

template <typename T>
PooledPtr<T>::~PooledPtr() noexcept {
    ReleaseIfLive();
}

}  // namespace sai::memory
