#include <sai/memory/arena_allocator.h>

#include <new>

#include <sai/memory/aligned_allocator.h>

namespace sai::memory {

ArenaAllocator::ArenaAllocator(std::size_t capacity_bytes) noexcept
    : region_(static_cast<std::byte*>(
          ::operator new(capacity_bytes, std::align_val_t{kSimdAlignment}))),
      capacity_bytes_(capacity_bytes),
      offset_(0) {}

ArenaAllocator::~ArenaAllocator() noexcept {
    ::operator delete(region_, std::align_val_t{kSimdAlignment});
}

auto ArenaAllocator::CapacityBytes() const noexcept -> std::size_t {
    return capacity_bytes_;
}

auto ArenaAllocator::UsedBytes() const noexcept -> std::size_t {
    return offset_.load(std::memory_order_relaxed);
}

}  // namespace sai::memory
