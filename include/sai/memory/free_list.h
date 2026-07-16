#pragma once

#include <atomic>
#include <cstdint>

namespace sai::memory {

// Lock-free CAS free-list operations (1.5-memory.md section 9).
//
// Template parameter Head is a TaggedHead-like struct with:
//   - .pointer  (Node*): head of the free-list stack
//   - .tag      (std::uint64_t): monotonically increasing tag to close the ABA window
//
// The Node type (inferred from Head::pointer) must have:
//   - .next  (Node*): pointer to the next free node
//
// These are header-only inline templates — the same instantiation compiles
// identically for GpuPool, PinnedPool, and HostTestPool.

// Single CAS retry loop, no nested branching.
// Returns the popped node, or nullptr if the list is empty.
template <typename Head>
[[nodiscard]] inline auto PopFreeList(std::atomic<Head>& head) noexcept
    -> decltype(Head{}.pointer) {
    Head old_head = head.load(std::memory_order_acquire);
    while (old_head.pointer != nullptr &&
           !head.compare_exchange_weak(
               old_head, Head{old_head.pointer->next, old_head.tag + 1},
               std::memory_order_acq_rel, std::memory_order_acquire)) {
        // old_head is refreshed to the latest observed {pointer, tag} by CAS; retry.
    }
    return old_head.pointer;
}

// Single CAS retry loop, no nested branching.
// Pushes node back onto the free-list stack.
template <typename Head>
inline void PushFreeList(std::atomic<Head>& head,
                         decltype(Head{}.pointer) node) noexcept {
    Head old_head = head.load(std::memory_order_acquire);
    Head new_head;
    do {
        node->next = old_head.pointer;
        new_head = Head{node, old_head.tag + 1};
    } while (!head.compare_exchange_weak(old_head, new_head,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire));
}

}  // namespace sai::memory
