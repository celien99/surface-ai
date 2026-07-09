#pragma once

#include <atomic>
#include <cstddef>
#include <source_location>
#include <utility>

#include <sai/core/error.h>

namespace sai::memory {

// Independent allocation source for pool metadata (free-list nodes, etc.),
// kept in an address range that never overlaps the business-data slab
// regions owned by GpuPool/PinnedPool/HostTestPool. Reserves capacity_bytes
// once from the OS at construction; runtime only carves up that region and
// never requests more from the OS. Exhaustion returns an error instead of
// falling back to heap allocation.
class ArenaAllocator final {
public:
    // capacity_bytes: total bytes this Arena reserves up front. The caller
    // (a pool's construction flow) precomputes this from slab_count and the
    // size of one free-list node before passing it in.
    explicit ArenaAllocator(std::size_t capacity_bytes) noexcept;
    ~ArenaAllocator() noexcept;

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    ArenaAllocator(ArenaAllocator&&) = delete;
    ArenaAllocator& operator=(ArenaAllocator&&) = delete;

    // Allocates storage for one T aligned to alignof(T) and constructs it in
    // place. The returned pointer lives as long as the Arena itself; callers
    // must not (and cannot) delete it individually — the Arena reclaims the
    // whole region on its own destruction. Returns Memory_ArenaExhausted once
    // capacity is used up.
    template <typename T, typename... Args>
    [[nodiscard]] auto Construct(Args&&... args) noexcept -> Result<T*>;

    [[nodiscard]] auto CapacityBytes() const noexcept -> std::size_t;
    [[nodiscard]] auto UsedBytes() const noexcept -> std::size_t;

private:
    std::byte* region_;
    std::size_t capacity_bytes_;
    std::atomic<std::size_t> offset_;  // Allocation cursor, advanced via CAS under concurrency.
};

template <typename T, typename... Args>
auto ArenaAllocator::Construct(Args&&... args) noexcept -> Result<T*> {
    constexpr std::size_t kAlignment = alignof(T);
    constexpr std::size_t kSize = sizeof(T);

    std::size_t old_offset = offset_.load(std::memory_order_relaxed);
    std::size_t aligned_offset = (old_offset + kAlignment - 1) & ~(kAlignment - 1);
    while (aligned_offset + kSize <= capacity_bytes_ &&
           !offset_.compare_exchange_weak(old_offset, aligned_offset + kSize,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
        aligned_offset = (old_offset + kAlignment - 1) & ~(kAlignment - 1);
    }

    if (aligned_offset + kSize > capacity_bytes_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_ArenaExhausted,
            "ArenaAllocator capacity exhausted",
            std::source_location::current(),
        });
    }

    return ::new (static_cast<void*>(region_ + aligned_offset)) T(std::forward<Args>(args)...);
}

}  // namespace sai::memory
