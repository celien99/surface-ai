#include "host_test_pool.h"

#include <source_location>

namespace sai::test {

using sai::memory::PooledPtr;

HostTestPool::HostTestPool(sai::memory::MemoryPoolConfig config) noexcept
    : config_(config),
      region_(config.slab_size * config.slab_count),
      nodes_(config.slab_count),
      available_count_(config.slab_count) {
    for (std::size_t i = 0; i < config.slab_count; ++i) {
        nodes_[i].slab_ptr = region_.data() + i * config.slab_size;
        nodes_[i].ref_count.store(0, std::memory_order_relaxed);
        nodes_[i].next = (i + 1 < config.slab_count) ? &nodes_[i + 1] : nullptr;
    }
    free_list_head_.store(TaggedHead{config.slab_count > 0 ? &nodes_[0] : nullptr, 0},
                           std::memory_order_relaxed);
}

HostTestPool::~HostTestPool() noexcept = default;

// Single CAS retry loop, no nested branching — reused verbatim from
// 1.5-memory.md §9's PopFreeList shape (tagged head, closes the ABA window).
auto HostTestPool::PopFreeList(std::atomic<TaggedHead>& head) noexcept -> Node* {
    TaggedHead old_head = head.load(std::memory_order_acquire);
    while (old_head.pointer != nullptr &&
           !head.compare_exchange_weak(
               old_head, TaggedHead{old_head.pointer->next, old_head.tag + 1},
               std::memory_order_acq_rel, std::memory_order_acquire)) {
        // old_head is refreshed to the latest observed {pointer, tag} by CAS; retry.
    }
    return old_head.pointer;
}

// Single CAS retry loop, no nested branching — reused verbatim from
// 1.5-memory.md §9's PushFreeList shape (tagged head, closes the ABA window).
void HostTestPool::PushFreeList(std::atomic<TaggedHead>& head, Node* node) noexcept {
    TaggedHead old_head = head.load(std::memory_order_acquire);
    TaggedHead new_head;
    do {
        node->next = old_head.pointer;
        new_head = TaggedHead{node, old_head.tag + 1};
    } while (!head.compare_exchange_weak(old_head, new_head, std::memory_order_acq_rel,
                                          std::memory_order_acquire));
}

auto HostTestPool::Acquire(std::size_t bytes) noexcept -> sai::Result<PooledPtr<std::uint8_t>> {
    if (bytes > config_.slab_size) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Memory_RequestExceedsSlabSize,
            "requested bytes exceed slab_size",
            std::source_location::current(),
        });
    }

    Node* node = PopFreeList(free_list_head_);
    if (node == nullptr) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Memory_PoolExhausted,
            "free list is empty",
            std::source_location::current(),
        });
    }

    available_count_.fetch_sub(1, std::memory_order_acq_rel);
    node->ref_count.store(1, std::memory_order_release);
    return MakeHandle(node->slab_ptr, config_.slab_size, this, &node->ref_count);
}

void HostTestPool::Release(PooledPtr<std::uint8_t>& handle) noexcept {
    std::uint8_t* slab_ptr = handle.Get();
    if (!DropReference(handle)) {
        return;
    }

    // The refcount slot lives inside the Node that owns this slab; recover
    // the Node from the slab pointer to push it back onto the free list.
    const std::size_t index = static_cast<std::size_t>(slab_ptr - region_.data()) / config_.slab_size;
    PushFreeList(free_list_head_, &nodes_[index]);
    available_count_.fetch_add(1, std::memory_order_acq_rel);
}

auto HostTestPool::SlabSize() const noexcept -> std::size_t {
    return config_.slab_size;
}

auto HostTestPool::SlabCount() const noexcept -> std::size_t {
    return config_.slab_count;
}

auto HostTestPool::AvailableSlabCount() const noexcept -> std::size_t {
    return available_count_.load(std::memory_order_acquire);
}

}  // namespace sai::test
