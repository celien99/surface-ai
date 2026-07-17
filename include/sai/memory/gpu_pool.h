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

// Device-memory pool wrapping cudaMalloc/cudaFree. Create() performs one
// cudaMalloc(slab_size * slab_count) up front and builds the initial free
// list; the destructor performs one cudaFree. Acquire/Release never call
// cudaMalloc/cudaFree after construction — see 1.5-memory.md §3/§9. 1.4
// batch's CUDA Stream GPU queue can pass the raw pointer returned by
// PooledPtr<uint8_t>::Get() directly to CUDA APIs (e.g. cudaMemcpyAsync) as
// the device-side address.
//
// CUDA-gated: this file is only compiled into sai_memory when
// find_package(CUDAToolkit) succeeds (see src/memory/CMakeLists.txt). It is
// not built, and not expected to build, on hosts without the CUDA Toolkit.
class GpuPool final : public IMemoryPool {
public:
    // config: slab_size/slab_count. arena: this pool's free-list nodes are
    // allocated from it (host memory metadata, never from the device region
    // itself — see 1.5-memory.md §11's metadata/business-data separation);
    // the caller must keep arena alive for at least as long as this pool.
    // Construction failure (cudaMalloc failure, arena exhaustion) is
    // reported through Result rather than an exception — this factory
    // exists specifically to avoid needing to express Result from a real
    // constructor, see 1.5-memory.md §4's rationale on GpuPool::Create.
    [[nodiscard]] static auto Create(MemoryPoolConfig config, ArenaAllocator& arena) noexcept
        -> Result<std::unique_ptr<GpuPool>>;

    ~GpuPool() noexcept override;

    GpuPool(const GpuPool&) = delete;
    GpuPool& operator=(const GpuPool&) = delete;
    GpuPool(GpuPool&&) = delete;
    GpuPool& operator=(GpuPool&&) = delete;

    [[nodiscard]] auto Acquire(std::size_t bytes) noexcept
        -> Result<PooledPtr<std::uint8_t>> override;
    void Release(PooledPtr<std::uint8_t>& handle) noexcept override;

    [[nodiscard]] auto SlabSize() const noexcept -> std::size_t override;
    [[nodiscard]] auto SlabCount() const noexcept -> std::size_t override;
    [[nodiscard]] auto AvailableSlabCount() const noexcept -> std::size_t override;

private:
    // One node per slab: next pointer for the lock-free free-list stack, the
    // slab's device-side data pointer, and the refcount slot PooledPtr
    // instances point at. Each Node is constructed one at a time via the
    // caller-supplied ArenaAllocator (arena.Construct<Node>()), per
    // 1.5-memory.md §11's metadata/business-data separation — Node storage
    // lives in the arena's host-memory region, never inside device_region_
    // and never freed individually; a node is always either on the free
    // list or held by some PooledPtr. slab_ptr points into device_region_
    // (device memory); host code never dereferences it directly, only
    // passes it to CUDA APIs or hands it out via PooledPtr.
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
    // GCC's std::atomic<T>::is_always_lock_free is a libstdc++ ABI constant that
    // reports false for 16-byte types even when -mcx16 generates correct CMPXCHG16B
    // instructions. Use the compiler intrinsic macro instead, which correctly reflects
    // whether the compiler can emit lock-free 16-byte CAS.
    static_assert(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16,
                  "TaggedHead must be lock-free per 1.5-memory.md §12 — "
                  "compile with -mcx16 (or -march=x86-64-v2+) to enable CMPXCHG16B");

    // PopFreeList / PushFreeList are provided by <sai/memory/free_list.h>
    // as inline template functions — see 1.5-memory.md section 9.

    GpuPool() noexcept = default;

    std::byte* device_region_ = nullptr;  // cudaMalloc'd device-side base address.
    MemoryPoolConfig config_{};
    std::atomic<TaggedHead> free_list_head_{};  // Lock-free stack head (tag closes the ABA window).

    // Not part of 1.5-memory.md §4's illustrative private-field list, but
    // required to implement Release() correctly: nodes_[i] is the
    // arena-constructed Node for slab i, letting Release() recover a Node
    // from the slab pointer it gets back (PooledPtr<uint8_t> only carries
    // the raw slab pointer, not the owning Node). Populated once during
    // Create() as each Node is constructed via arena.Construct<Node>(); the
    // vector itself is a one-time Create()-time allocation, not part of the
    // Acquire/Release hot path's "zero runtime dynamic allocation" budget —
    // the Node objects it points at live in arena storage, this vector only
    // holds the pointers needed for the slab_ptr -> Node lookup.
    std::vector<Node*> nodes_;

    // Also not in §4's illustrative list, for the same reason HostTestPool
    // needed one: walking the lock-free list to count free slabs would be
    // racy under concurrent Push/Pop, so AvailableSlabCount() is tracked via
    // its own atomic counter updated alongside every successful Acquire/
    // Release, exactly as tests/memory/host_test_pool.cpp does.
    std::atomic<std::size_t> available_count_{0};
};

}  // namespace sai::memory
