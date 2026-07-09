#pragma once

#include <cstddef>
#include <new>

namespace sai::memory {

inline constexpr std::size_t kSimdAlignment = 64;  // AVX-512 cache line width, fixed globally, not configurable.

// Used for standalone allocations that need SIMD alignment (e.g. Arena
// Allocator's own region request); slabs internally round up to
// kSimdAlignment so every slab's start address is aligned.
template <typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    [[nodiscard]] auto allocate(std::size_t n) -> T* {
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{kSimdAlignment}));
    }

    void deallocate(T* ptr, std::size_t /*n*/) noexcept {
        ::operator delete(ptr, std::align_val_t{kSimdAlignment});
    }
};

template <typename T, typename U>
auto operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept -> bool {
    return true;
}

template <typename T, typename U>
auto operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept -> bool {
    return false;
}

}  // namespace sai::memory
