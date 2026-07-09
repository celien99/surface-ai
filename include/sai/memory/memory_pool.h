#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/memory/pooled_ptr.h>

namespace sai::memory {

struct MemoryPoolConfig {
    std::size_t slab_size;   // Bytes per slab; requests over this fail outright, no multi-slab stitching.
    std::size_t slab_count;  // Slabs pre-allocated at startup, usually derived from max_concurrent_frames.
};

// The unified pooled-allocation contract. GpuPool/PinnedPool are this
// interface's concrete implementations for the two physical memory kinds;
// neither calls cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost outside of
// Acquire/Release.
class IMemoryPool : public Object {
public:
    ~IMemoryPool() override = default;

    // Pops one slab off the free list and wraps it as a PooledPtr<uint8_t>.
    // bytes above the slab_size agreed at construction returns
    // Memory_RequestExceedsSlabSize; an empty free list (every slab_count
    // slab in use) returns Memory_PoolExhausted — no blocking wait, no
    // fallback to dynamic allocation. The caller (Capture/Inference thread)
    // decides whether to retry, drop the frame, or propagate the error; the
    // pool does not make that call on the caller's behalf.
    [[nodiscard]] virtual auto Acquire(std::size_t bytes) noexcept
        -> Result<PooledPtr<uint8_t>> = 0;

    // Explicit return: decrements the refcount, and on reaching zero, pushes
    // the slab back onto the free list. The handle is cleared (data_ =
    // nullptr) afterwards so a returned handle can't be misused; normally
    // this doesn't need to be called by hand — PooledPtr<T>'s destructor
    // performs the exact same operation automatically. Release only exists
    // for call sites that need to return a slab ahead of the handle's own
    // scope exit.
    virtual void Release(PooledPtr<uint8_t>& handle) noexcept = 0;

    [[nodiscard]] virtual auto SlabSize() const noexcept -> std::size_t = 0;
    [[nodiscard]] virtual auto SlabCount() const noexcept -> std::size_t = 0;
    [[nodiscard]] virtual auto AvailableSlabCount() const noexcept -> std::size_t = 0;

protected:
    // PooledPtr<T>'s constructor is private and grants friendship to
    // IMemoryPool only, not to whichever concrete class derives from it —
    // C++ does not propagate a friend grant down an inheritance chain. This
    // factory lives on the base class specifically so GpuPool/PinnedPool/
    // HostTestPool (and any future IMemoryPool implementation) can build
    // handles without each needing its own friend declaration inside
    // pooled_ptr.h. Not part of the frozen public contract in §4 — purely an
    // implementation seam.
    template <typename T>
    static auto MakeHandle(T* data, std::size_t size_bytes, IMemoryPool* owner_pool,
                            std::atomic<int>* ref_count) noexcept -> PooledPtr<T> {
        return PooledPtr<T>(data, size_bytes, owner_pool, ref_count);
    }

    // Shared "decrement, then clear the handle" half of the return protocol,
    // usable both by a concrete pool's Release() override and by
    // PooledPtr<uint8_t>'s own destructor (indirectly, through Release()).
    // Returns true when this was the last outstanding reference, telling the
    // caller it must push the underlying slab back onto its own free list —
    // that half is necessarily pool-specific (each concrete pool owns a
    // different free-list representation) and is left to the caller.
    template <typename T>
    static auto DropReference(PooledPtr<T>& handle) noexcept -> bool {
        if (handle.data_ == nullptr) {
            return false;
        }
        bool const was_last_reference =
            handle.ref_count_->fetch_sub(1, std::memory_order_acq_rel) == 1;
        handle.data_ = nullptr;
        handle.size_bytes_ = 0;
        handle.owner_pool_ = nullptr;
        handle.ref_count_ = nullptr;
        return was_last_reference;
    }
};

// PooledPtr<T>'s destructor and assignment operators call ReleaseIfLive(),
// which is declared in pooled_ptr.h but defined here: it dispatches through
// the public Release() virtual, which requires IMemoryPool to be a complete
// type.
template <typename T>
void PooledPtr<T>::ReleaseIfLive() noexcept {
    if (data_ == nullptr) {
        return;
    }
    owner_pool_->Release(*this);
}

}  // namespace sai::memory
