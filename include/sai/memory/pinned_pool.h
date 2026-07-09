#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/memory/arena_allocator.h>
#include <sai/memory/memory_pool.h>
#include <sai/memory/pooled_ptr.h>

namespace sai::memory {

// Pinned host-memory pool wrapping cudaHostAlloc/cudaFreeHost. Create()
// performs one cudaHostAlloc(slab_size * slab_count, cudaHostAllocDefault)
// up front and builds the initial free list; the destructor performs one
// cudaFreeHost. Acquire/Release never call cudaHostAlloc/cudaFreeHost after
// construction — see 1.5-memory.md §3/§9. Used for host<->device
// cudaMemcpyAsync intermediate buffers, since ordinary heap memory is not
// page-locked and cannot back asynchronous transfers.
//
// CUDA-gated: this file is only compiled into sai_memory when
// find_package(CUDAToolkit) succeeds (see src/memory/CMakeLists.txt). It is
// not built, and not expected to build, on hosts without the CUDA Toolkit.
// Task 7's GpuStreamQueue takes a PinnedPool& in its constructor.
class PinnedPool final : public IMemoryPool {
public:
    // config: slab_size/slab_count. arena: this pool's free-list nodes are
    // allocated from it (host memory metadata, physically separate from the
    // pinned region itself — see 1.5-memory.md §11); the caller must keep
    // arena alive for at least as long as this pool. Construction failure
    // (cudaHostAlloc failure, arena exhaustion) is reported through Result
    // rather than an exception, per 1.5-memory.md §4.
    [[nodiscard]] static auto Create(MemoryPoolConfig config, ArenaAllocator& arena) noexcept
        -> Result<std::unique_ptr<PinnedPool>>;

    ~PinnedPool() noexcept override;

    PinnedPool(const PinnedPool&) = delete;
    PinnedPool& operator=(const PinnedPool&) = delete;
    PinnedPool(PinnedPool&&) = delete;
    PinnedPool& operator=(PinnedPool&&) = delete;

    [[nodiscard]] auto Acquire(std::size_t bytes) noexcept
        -> Result<PooledPtr<std::uint8_t>> override;
    void Release(PooledPtr<std::uint8_t>& handle) noexcept override;

    [[nodiscard]] auto SlabSize() const noexcept -> std::size_t override;
    [[nodiscard]] auto SlabCount() const noexcept -> std::size_t override;
    [[nodiscard]] auto AvailableSlabCount() const noexcept -> std::size_t override;

private:
    // One node per slab: next pointer for the lock-free free-list stack, the
    // slab's pinned-host data pointer, and the refcount slot PooledPtr
    // instances point at. Each Node is constructed one at a time via the
    // caller-supplied ArenaAllocator (arena.Construct<Node>()), per
    // 1.5-memory.md §11's metadata/business-data separation — Node storage
    // lives in the arena's region, never inside host_region_, and never
    // freed individually; a node is always either on the free list or held
    // by some PooledPtr.
    struct Node {
        Node* next;
        std::uint8_t* slab_ptr;
        std::atomic<int> ref_count;
    };

    // Head pointer plus a monotonically increasing tag, per 1.5-memory.md
    // §9: closes the ABA window a bare Node* head would leave open. Mirrors
    // tests/memory/host_test_pool.h's TaggedHead exactly.
    struct TaggedHead {
        Node* pointer = nullptr;
        std::uint64_t tag = 0;
    };

    // 1.5-memory.md §12: this scheme must be genuinely lock-free, with no
    // silent runtime fallback to a mutex.
    static_assert(std::atomic<TaggedHead>::is_always_lock_free,
                  "TaggedHead must be lock-free per 1.5-memory.md §12 — no silent fallback to a mutex");

    // Single CAS retry loop, no nested branching — reused verbatim from
    // 1.5-memory.md §9's PopFreeList/PushFreeList shape (same algorithm
    // tests/memory/host_test_pool.cpp already implements and tests).
    static auto PopFreeList(std::atomic<TaggedHead>& head) noexcept -> Node*;
    static void PushFreeList(std::atomic<TaggedHead>& head, Node* node) noexcept;

    PinnedPool() noexcept = default;

    std::byte* host_region_ = nullptr;  // cudaHostAlloc'd pinned host-side base address.
    MemoryPoolConfig config_{};
    std::atomic<TaggedHead> free_list_head_{};  // Lock-free stack head (tag closes the ABA window).

    // See GpuPool's identical field for the full rationale: recovers the
    // owning Node from a returned slab pointer inside Release(), and is a
    // one-time Create()-time allocation, not a hot-path allocation.
    std::vector<Node*> nodes_;

    // See GpuPool's identical field: walking the lock-free list to count
    // free slabs would be racy under concurrent Push/Pop, so this is
    // tracked via its own atomic counter, exactly as
    // tests/memory/host_test_pool.cpp does.
    std::atomic<std::size_t> available_count_{0};
};

}  // namespace sai::memory
