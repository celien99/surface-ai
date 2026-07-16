#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sai/core/error.h>
#include <sai/memory/memory_pool.h>
#include <sai/memory/pooled_ptr.h>

namespace sai::test {

// Test-only IMemoryPool implementation backed by plain std::malloc'd host
// memory, standing in for the real GpuPool/PinnedPool (Task 2's CUDA-gated
// deliverable) so PooledPtr<uint8_t>'s reference-counting and pool-return
// behavior can be exercised against a real pool on this machine. Uses the
// same slab + lock-free free-list design 1.5-memory.md §9 specifies for the
// real pools, just without cudaMalloc. Lives under tests/memory/ only — not
// a public library type, not installed, not part of include/.
class HostTestPool final : public sai::memory::IMemoryPool {
public:
    explicit HostTestPool(sai::memory::MemoryPoolConfig config) noexcept;
    ~HostTestPool() noexcept override;

    HostTestPool(const HostTestPool&) = delete;
    HostTestPool& operator=(const HostTestPool&) = delete;
    HostTestPool(HostTestPool&&) = delete;
    HostTestPool& operator=(HostTestPool&&) = delete;

    [[nodiscard]] auto Acquire(std::size_t bytes) noexcept
        -> sai::Result<sai::memory::PooledPtr<std::uint8_t>> override;
    void Release(sai::memory::PooledPtr<std::uint8_t>& handle) noexcept override;

    [[nodiscard]] auto SlabSize() const noexcept -> std::size_t override;
    [[nodiscard]] auto SlabCount() const noexcept -> std::size_t override;
    [[nodiscard]] auto AvailableSlabCount() const noexcept -> std::size_t override;

private:
    // One node per slab: next pointer for the lock-free free-list stack,
    // the slab's data pointer, and the refcount slot PooledPtr instances
    // point at. Pre-allocated at construction time, never freed individually
    // — a node is always either on the free list or held by some PooledPtr.
    struct Node {
        Node* next;
        std::uint8_t* slab_ptr;
        std::atomic<int> ref_count;
    };

    // Head pointer plus a monotonically increasing tag, per 1.5-memory.md §9:
    // a bare atomic<Node*> head is vulnerable to ABA (a thread's stale cached
    // `next` can be mistaken for still-valid state after other threads pop
    // and re-push the same node). Packing a tag alongside the pointer into
    // one atomically-compared 16-byte struct closes that window — any
    // intervening push/pop changes the tag, so the CAS fails and retries
    // instead of committing on a coincidentally-matching pointer value.
    struct TaggedHead {
        Node* pointer = nullptr;
        std::uint64_t tag = 0;
    };

    // 1.5-memory.md §12: this scheme must be genuinely lock-free on the
    // target platforms, with no silent runtime fallback to an internal
    // mutex. is_always_lock_free is a compile-time guarantee (unlike the
    // runtime is_lock_free() check on one instance) that every instance of
    // atomic<TaggedHead> on this platform/ABI compiles to a real lock-free
    // CAS, so a toolchain/ABI change that would break that guarantee fails
    // the build instead of silently degrading.
    static_assert(std::atomic<TaggedHead>::is_always_lock_free,
                  "TaggedHead must be lock-free per 1.5-memory.md §12 — no silent fallback to a mutex");

    // PopFreeList / PushFreeList are provided by <sai/memory/free_list.h>
    // as inline template functions -- see 1.5-memory.md section 9.

    sai::memory::MemoryPoolConfig config_;
    std::vector<std::uint8_t> region_;  // std::malloc'd stand-in for cudaMalloc'd device memory.
    std::vector<Node> nodes_;
    std::atomic<TaggedHead> free_list_head_{};
    std::atomic<std::size_t> available_count_;
};

}  // namespace sai::test
