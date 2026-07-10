#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/gpu_stream_queue.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------
//
// GpuStreamQueue is the GPU-task submission entry point: a pool of
// cudaStream_t plus one dedicated GPU Callback Thread. A host<->device
// async copy is issued with cudaMemcpyAsync + cudaStreamAddCallback; the
// driver-thread callback does nothing but push a GpuCompletionEvent into a
// bounded lock-free MPSC queue and return immediately — the single GPU
// Callback Thread pops that event and calls resume() (see §3/§9). No
// stop_callback is registered for the in-flight duration: the completion
// callback is the ONLY caller allowed to resume the coroutine handle, which
// structurally rules out the double-resume() a cancellation callback would
// otherwise race against (see §3).
//
// CUDA-gated: this header pulls in <cuda_runtime.h> and the translation unit
// gpu_stream_queue.cpp is only compiled into sai_runtime when
// find_package(CUDAToolkit) succeeds (see src/runtime/CMakeLists.txt). It is
// not built, and not expected to build, on hosts without the CUDA Toolkit
// (e.g. the macOS arm64 dev machine); nothing in the portable test suite
// includes this header.

#include <atomic>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <sai/core/error.h>
#include <sai/memory/pinned_pool.h>
#include <sai/memory/pooled_ptr.h>
#include <sai/runtime/task.h>

namespace sai::runtime {

// §4 freezes these names unqualified inside sai::runtime; PinnedPool and
// PooledPtr are owned by batch 1.5 and referenced verbatim, never redefined.
using ::sai::memory::PinnedPool;
using ::sai::memory::PooledPtr;

// GPU completion event forwarded from a CUDA driver thread to the GPU Callback
// Thread: the coroutine handle to resume and the operation's terminal status.
// Value-typed; it lives in the MPSC queue only between push and pop.
struct GpuCompletionEvent {
    std::coroutine_handle<> handle;
    cudaError_t status;
};

// Direction of a host<->device async copy. Only the two directions the
// framework actually needs (Capture host->device, Inference device->host);
// DeviceToDevice is intentionally absent (see §4/§12).
enum class CopyDirection {
    HostToDevice,
    DeviceToHost,
};

class GpuStreamQueue final {
public:
    // stream_count: number of cudaStream_t submitted GPU tasks are assigned
    // across round-robin. pinned_pool: source of the host-side transfer buffer
    // for every copy; the caller keeps it alive for this queue's whole life.
    // Construction failure (cudaStreamCreate) is reported through Result.
    [[nodiscard]] static auto Create(std::size_t stream_count, PinnedPool& pinned_pool) noexcept
        -> Result<std::unique_ptr<GpuStreamQueue>>;

    ~GpuStreamQueue() noexcept;

    GpuStreamQueue(const GpuStreamQueue&) = delete;
    GpuStreamQueue& operator=(const GpuStreamQueue&) = delete;
    GpuStreamQueue(GpuStreamQueue&&) = delete;
    GpuStreamQueue& operator=(GpuStreamQueue&&) = delete;

    // stop_token is read exactly once, at entry (checkpoint 1): if a stop is
    // already requested the copy is not issued and Runtime_Cancelled is
    // returned. Once cudaMemcpyAsync + cudaStreamAddCallback are issued, no
    // stop_callback is registered for the flight — the in-flight op is
    // cancellation-immune and the completion callback is its single resumer
    // (see §3). The returned Task<void> resumes only after the GPU op finishes
    // and the completion is forwarded through the GPU Callback Thread.
    [[nodiscard]] auto EnqueueAsyncCopy(PooledPtr<std::uint8_t> device_or_host_src,
                                        std::size_t bytes,
                                        CopyDirection direction,
                                        std::stop_token stop_token) noexcept -> Task<void>;

    [[nodiscard]] auto StreamCount() const noexcept -> std::size_t;

private:
    GpuStreamQueue() noexcept = default;

    // Bounded lock-free MPSC ring (Vyukov bounded queue, valid for the single
    // consumer we use): producers are CUDA driver callback threads, the sole
    // consumer is gpu_callback_thread_. The push path is a CAS with no lock, so
    // a driver thread never blocks on it (see §9). ticket_ is a wait/notify
    // counter so the consumer parks with zero CPU when the queue is empty and
    // is woken by each push (and by WakeAll() at shutdown) rather than busy
    // spinning.
    class CompletionQueue {
    public:
        void Init(std::size_t capacity) noexcept {
            std::size_t cap = 2;
            while (cap < capacity) {
                cap <<= 1U;
            }
            buffer_ = std::vector<Cell>(cap);
            mask_ = cap - 1;
            for (std::size_t i = 0; i < cap; ++i) {
                buffer_[i].sequence.store(i, std::memory_order_relaxed);
            }
            enqueue_pos_.store(0, std::memory_order_relaxed);
            dequeue_pos_.store(0, std::memory_order_relaxed);
            ticket_.store(0, std::memory_order_relaxed);
        }

        // Returns false only when full; with the §11 capacity invariant
        // (>= PinnedPool::SlabCount()) that cannot happen at runtime.
        bool Push(const GpuCompletionEvent& event) noexcept {
            std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
            Cell* cell = nullptr;
            for (;;) {
                cell = &buffer_[pos & mask_];
                const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
                const auto dif =
                    static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
                if (dif == 0) {
                    if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                                           std::memory_order_relaxed)) {
                        break;
                    }
                } else if (dif < 0) {
                    return false;
                } else {
                    pos = enqueue_pos_.load(std::memory_order_relaxed);
                }
            }
            cell->data = event;
            cell->sequence.store(pos + 1, std::memory_order_release);
            ticket_.fetch_add(1, std::memory_order_release);
            ticket_.notify_one();
            return true;
        }

        bool Pop(GpuCompletionEvent& out) noexcept {
            std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
            Cell* cell = nullptr;
            for (;;) {
                cell = &buffer_[pos & mask_];
                const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
                const auto dif =
                    static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
                if (dif == 0) {
                    if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                                           std::memory_order_relaxed)) {
                        break;
                    }
                } else if (dif < 0) {
                    return false;
                } else {
                    pos = dequeue_pos_.load(std::memory_order_relaxed);
                }
            }
            out = cell->data;
            cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
            return true;
        }

        [[nodiscard]] std::uint64_t Ticket() const noexcept {
            return ticket_.load(std::memory_order_acquire);
        }
        void Wait(std::uint64_t expected) noexcept {
            ticket_.wait(expected, std::memory_order_acquire);
        }
        void WakeAll() noexcept {
            ticket_.fetch_add(1, std::memory_order_release);
            ticket_.notify_all();
        }

    private:
        struct Cell {
            std::atomic<std::size_t> sequence;
            GpuCompletionEvent data{};
        };
        std::vector<Cell> buffer_;
        std::size_t mask_ = 0;
        std::atomic<std::size_t> enqueue_pos_{0};
        std::atomic<std::size_t> dequeue_pos_{0};
        std::atomic<std::uint64_t> ticket_{0};
    };

    // Awaiter that issues the copy and suspends until the completion is
    // forwarded back; defined in gpu_stream_queue.cpp. Nested so it can touch
    // this queue's private streams_/next_stream_/completion_queue_.
    struct GpuCopyAwaiter;

    // cudaStreamAddCallback trampoline: runs on a CUDA driver thread, only
    // pushes a GpuCompletionEvent and returns — never resume() here (§3/§14).
    static void CUDART_CB OnStreamDone(cudaStream_t stream, cudaError_t status,
                                       void* user_data);

    // gpu_callback_thread_ body: pop completions and resume(), parking on the
    // queue's ticket wait when idle. The single GPU Callback Thread (§9).
    void CallbackLoop(std::stop_token stop_token) noexcept;

    std::vector<cudaStream_t> streams_;
    PinnedPool* pinned_pool_ = nullptr;          // Borrowed; does not own its lifetime.
    std::atomic<std::size_t> next_stream_{0};    // Round-robin stream selector.
    CompletionQueue completion_queue_;           // MPSC forwarding queue (§9).
    std::jthread gpu_callback_thread_;           // Declared last: joined first at teardown.
};

}  // namespace sai::runtime
