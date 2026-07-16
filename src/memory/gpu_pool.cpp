#include <sai/memory/gpu_pool.h>

#include <cstdint>
#include <source_location>
#include <string>

#include <cuda_runtime.h>
#include <sai/memory/free_list.h>

namespace sai::memory {

auto GpuPool::Create(MemoryPoolConfig config, ArenaAllocator& arena) noexcept
    -> Result<std::unique_ptr<GpuPool>> {
    auto pool = std::unique_ptr<GpuPool>(new GpuPool());
    pool->config_ = config;

    void* raw_device_ptr = nullptr;
    const cudaError_t malloc_status =
        cudaMalloc(&raw_device_ptr, config.slab_size * config.slab_count);
    if (malloc_status != cudaSuccess) {
        // No dedicated Memory_* code exists for a driver allocation failure
        // at construction time (only Memory_ArenaExhausted/
        // Memory_RequestExceedsSlabSize/Memory_PoolExhausted are defined so
        // far, and this task's scope does not add a new one — see the
        // report for the reasoning). Memory_PoolExhausted is reused: from
        // the caller's perspective "cudaMalloc failed, so this pool never
        // has any slabs to give out" and "the free list ran dry" are the
        // same observable outcome (Acquire can never succeed), even though
        // the underlying cause differs.
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_PoolExhausted,
            std::string("cudaMalloc failed: ") + cudaGetErrorString(malloc_status),
            std::source_location::current(),
        });
    }
    pool->device_region_ = static_cast<std::byte*>(raw_device_ptr);

    pool->nodes_.reserve(config.slab_count);
    Node* first = nullptr;
    Node* previous = nullptr;
    for (std::size_t i = 0; i < config.slab_count; ++i) {
        auto node_result = arena.Construct<Node>();
        if (!node_result.has_value()) {
            // Don't cudaFree here: pool (a unique_ptr<GpuPool>) is about to
            // go out of scope on this return, and ~GpuPool() already frees
            // device_region_ exactly once. Freeing it here too and leaving
            // device_region_ non-null would double-free it in the
            // destructor.
            return tl::make_unexpected(node_result.error());
        }

        Node* node = *node_result;
        node->slab_ptr = reinterpret_cast<std::uint8_t*>(pool->device_region_) + i * config.slab_size;
        node->ref_count.store(0, std::memory_order_relaxed);
        node->next = nullptr;
        pool->nodes_.push_back(node);

        if (previous == nullptr) {
            first = node;
        } else {
            previous->next = node;
        }
        previous = node;
    }

    pool->free_list_head_.store(TaggedHead{first, 0}, std::memory_order_relaxed);
    pool->available_count_.store(config.slab_count, std::memory_order_relaxed);

    return pool;
}

GpuPool::~GpuPool() noexcept {
    if (device_region_ != nullptr) {
        cudaFree(device_region_);
    }
}

auto GpuPool::Acquire(std::size_t bytes) noexcept -> Result<PooledPtr<std::uint8_t>> {
    if (bytes > config_.slab_size) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_RequestExceedsSlabSize,
            "requested bytes exceed slab_size",
            std::source_location::current(),
        });
    }

    Node* node = PopFreeList(free_list_head_);
    if (node == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_PoolExhausted,
            "free list is empty",
            std::source_location::current(),
        });
    }

    available_count_.fetch_sub(1, std::memory_order_acq_rel);
    node->ref_count.store(1, std::memory_order_release);
    return MakeHandle(node->slab_ptr, config_.slab_size, this, &node->ref_count);
}

void GpuPool::Release(PooledPtr<std::uint8_t>& handle) noexcept {
    std::uint8_t* slab_ptr = handle.Get();
    if (!DropReference(handle)) {
        return;
    }

    // slab_ptr and device_region_ are both device-memory addresses; this is
    // pointer arithmetic on address values only (no host dereference of
    // either pointer), used purely to recover which Node owns this slab —
    // the same index recovery HostTestPool performs against its host
    // region_.
    const std::size_t index =
        static_cast<std::size_t>(slab_ptr - reinterpret_cast<std::uint8_t*>(device_region_)) /
        config_.slab_size;
    PushFreeList(free_list_head_, nodes_[index]);
    available_count_.fetch_add(1, std::memory_order_acq_rel);
}

auto GpuPool::SlabSize() const noexcept -> std::size_t {
    return config_.slab_size;
}

auto GpuPool::SlabCount() const noexcept -> std::size_t {
    return config_.slab_count;
}

auto GpuPool::AvailableSlabCount() const noexcept -> std::size_t {
    return available_count_.load(std::memory_order_acquire);
}

}  // namespace sai::memory
