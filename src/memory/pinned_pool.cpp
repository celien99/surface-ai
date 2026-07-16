#include <sai/memory/pinned_pool.h>

#include <cstdint>
#include <source_location>
#include <string>

#include <cuda_runtime.h>
#include <sai/memory/free_list.h>

namespace sai::memory {

auto PinnedPool::Create(MemoryPoolConfig config, ArenaAllocator& arena) noexcept
    -> Result<std::unique_ptr<PinnedPool>> {
    auto pool = std::unique_ptr<PinnedPool>(new PinnedPool());
    pool->config_ = config;

    void* raw_host_ptr = nullptr;
    const cudaError_t alloc_status = cudaHostAlloc(
        &raw_host_ptr, config.slab_size * config.slab_count, cudaHostAllocDefault);
    if (alloc_status != cudaSuccess) {
        // Same reasoning as GpuPool::Create: no dedicated Memory_* code
        // exists yet for a driver allocation failure at construction time,
        // so Memory_PoolExhausted is reused — from the caller's
        // perspective, "cudaHostAlloc failed" and "the free list is
        // permanently empty" are the same observable outcome (Acquire can
        // never succeed). See the report for the full reasoning.
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_PoolExhausted,
            std::string("cudaHostAlloc failed: ") + cudaGetErrorString(alloc_status),
            std::source_location::current(),
        });
    }
    pool->host_region_ = static_cast<std::byte*>(raw_host_ptr);

    pool->nodes_.reserve(config.slab_count);
    Node* first = nullptr;
    Node* previous = nullptr;
    for (std::size_t i = 0; i < config.slab_count; ++i) {
        auto node_result = arena.Construct<Node>();
        if (!node_result.has_value()) {
            // Don't cudaFreeHost here: pool (a unique_ptr<PinnedPool>) is
            // about to go out of scope on this return, and ~PinnedPool()
            // already frees host_region_ exactly once. Freeing it here too
            // and leaving host_region_ non-null would double-free it in the
            // destructor.
            return tl::make_unexpected(node_result.error());
        }

        Node* node = *node_result;
        node->slab_ptr = reinterpret_cast<std::uint8_t*>(pool->host_region_) + i * config.slab_size;
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

PinnedPool::~PinnedPool() noexcept {
    if (host_region_ != nullptr) {
        cudaFreeHost(host_region_);
    }
}

auto PinnedPool::Acquire(std::size_t bytes) noexcept -> Result<PooledPtr<std::uint8_t>> {
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

void PinnedPool::Release(PooledPtr<std::uint8_t>& handle) noexcept {
    std::uint8_t* slab_ptr = handle.Get();
    if (!DropReference(handle)) {
        return;
    }

    const std::size_t index =
        static_cast<std::size_t>(slab_ptr - reinterpret_cast<std::uint8_t*>(host_region_)) /
        config_.slab_size;
    PushFreeList(free_list_head_, nodes_[index]);
    available_count_.fetch_add(1, std::memory_order_acq_rel);
}

auto PinnedPool::SlabSize() const noexcept -> std::size_t {
    return config_.slab_size;
}

auto PinnedPool::SlabCount() const noexcept -> std::size_t {
    return config_.slab_count;
}

auto PinnedPool::AvailableSlabCount() const noexcept -> std::size_t {
    return available_count_.load(std::memory_order_acquire);
}

}  // namespace sai::memory
