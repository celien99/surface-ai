#include <sai/runtime/gpu_stream_queue.h>

#include <cstdint>
#include <source_location>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace sai::runtime {

// Awaiter for a single host<->device async copy. Lives inside the
// EnqueueAsyncCopy coroutine frame for the whole suspension, so its address is
// a stable userData handed to cudaStreamAddCallback and OnStreamDone can write
// back into it. As a nested type it reaches GpuStreamQueue's private members.
struct GpuStreamQueue::GpuCopyAwaiter {
    GpuStreamQueue* self;
    void* device_ptr;
    void* pinned_ptr;
    std::size_t bytes;
    CopyDirection direction;
    std::coroutine_handle<> handle{};
    cudaError_t status{cudaSuccess};

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<> awaiting) noexcept {
        handle = awaiting;

        const std::size_t idx =
            self->next_stream_.fetch_add(1, std::memory_order_relaxed) % self->streams_.size();
        cudaStream_t stream = self->streams_[idx];

        void* dst = direction == CopyDirection::HostToDevice ? device_ptr : pinned_ptr;
        void* src = direction == CopyDirection::HostToDevice ? pinned_ptr : device_ptr;
        const cudaMemcpyKind kind = direction == CopyDirection::HostToDevice
                                        ? cudaMemcpyHostToDevice
                                        : cudaMemcpyDeviceToHost;

        status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (status != cudaSuccess) {
            // The copy never launched: no callback will fire, so don't suspend —
            // resume immediately and surface the error via await_resume().
            return false;
        }

        status = cudaStreamAddCallback(stream, &GpuStreamQueue::OnStreamDone, this, 0);
        if (status != cudaSuccess) {
            // Rare: the copy is already in flight but no completion callback was
            // registered. Drain the stream so device_ptr/pinned_ptr stay valid
            // until the transfer finishes, then resume without suspending. This
            // is the only place that blocks, and only on an error path.
            cudaStreamSynchronize(stream);
            return false;
        }

        // Suspended: OnStreamDone (driver thread) will push a completion event,
        // and the GPU Callback Thread will resume this handle. No stop_callback
        // is registered for the flight — single resumer, see §3.
        return true;
    }

    [[nodiscard]] cudaError_t await_resume() const noexcept { return status; }
};

void CUDART_CB GpuStreamQueue::OnStreamDone(cudaStream_t /*stream*/, cudaError_t status,
                                            void* user_data) {
    auto* awaiter = static_cast<GpuCopyAwaiter*>(user_data);
    awaiter->status = status;
    // Only action on the driver thread: a single lock-free push, then return.
    // The GPU Callback Thread does the resume() (§3/§9/§14).
    awaiter->self->completion_queue_.Push(GpuCompletionEvent{awaiter->handle, status});
}

void GpuStreamQueue::CallbackLoop(std::stop_token stop_token) noexcept {
    // Waking the ticket on stop lets a parked Wait() return so the loop can
    // observe stop_requested() and exit.
    const std::stop_callback wake(stop_token, [this] { completion_queue_.WakeAll(); });

    while (!stop_token.stop_requested()) {
        const std::uint64_t before = completion_queue_.Ticket();
        GpuCompletionEvent event;
        if (completion_queue_.Pop(event)) {
            if (event.handle) {
                event.handle.resume();
            }
            continue;
        }
        // Empty: park until a push (or WakeAll) advances the ticket past the
        // value observed before the failed Pop, so no completion is missed.
        completion_queue_.Wait(before);
    }

    // Drain completions queued before stop so no coroutine is left unresumed.
    GpuCompletionEvent event;
    while (completion_queue_.Pop(event)) {
        if (event.handle) {
            event.handle.resume();
        }
    }
}

auto GpuStreamQueue::Create(std::size_t stream_count, PinnedPool& pinned_pool) noexcept
    -> Result<std::unique_ptr<GpuStreamQueue>> {
    auto queue = std::unique_ptr<GpuStreamQueue>(new GpuStreamQueue());
    queue->pinned_pool_ = &pinned_pool;

    queue->streams_.reserve(stream_count);
    for (std::size_t i = 0; i < stream_count; ++i) {
        cudaStream_t stream = nullptr;
        const cudaError_t status = cudaStreamCreate(&stream);
        if (status != cudaSuccess) {
            // Destroy the streams already created and clear the vector so the
            // discarded queue's destructor doesn't touch them again, then
            // report construction failure (Create() building the object failed).
            for (cudaStream_t created : queue->streams_) {
                cudaStreamDestroy(created);
            }
            queue->streams_.clear();
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Core_ConstructionFailed,
                std::string("cudaStreamCreate failed: ") + cudaGetErrorString(status),
                std::source_location::current(),
            });
        }
        queue->streams_.push_back(stream);
    }

    // At most SlabCount() copies can be in flight at once (each holds one pinned
    // transfer slab), so a queue sized past that can never reject a driver push;
    // Push() therefore never fails at runtime (§11).
    queue->completion_queue_.Init(pinned_pool.SlabCount() + 1);

    // Start the single GPU Callback Thread; its stop_token comes from the
    // jthread itself and is honored by CallbackLoop.
    queue->gpu_callback_thread_ =
        std::jthread([raw = queue.get()](std::stop_token stop_token) {
            raw->CallbackLoop(std::move(stop_token));
        });

    return queue;
}

GpuStreamQueue::~GpuStreamQueue() noexcept {
    // Stop and wake the callback thread first (WakeAll() releases a parked
    // Wait()), join it, then destroy the streams it may have been draining.
    gpu_callback_thread_.request_stop();
    completion_queue_.WakeAll();
    if (gpu_callback_thread_.joinable()) {
        gpu_callback_thread_.join();
    }
    for (cudaStream_t stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

auto GpuStreamQueue::EnqueueAsyncCopy(PooledPtr<std::uint8_t> device_or_host_src,
                                      std::size_t bytes, CopyDirection direction,
                                      std::stop_token stop_token) noexcept -> Task<void> {
    // Checkpoint 1 (before issue): cancellation is observed only here. If a stop
    // is already requested, don't issue the copy — see §3/§5.
    if (stop_token.stop_requested()) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_Cancelled,
            "EnqueueAsyncCopy cancelled before issuing cudaMemcpyAsync",
            std::source_location::current(),
        });
    }

    auto transfer = pinned_pool_->Acquire(bytes);
    if (!transfer) {
        co_return tl::make_unexpected(transfer.error());
    }

    // device_or_host_src and *transfer both live in this coroutine frame across
    // the suspension below, so neither slab is released while the copy is in
    // flight (§11). No stop_callback is registered for the flight; the GPU
    // completion callback is the single resumer of this coroutine (§3).
    const cudaError_t status = co_await GpuCopyAwaiter{
        this, device_or_host_src.Get(), transfer->Get(), bytes, direction};

    // Checkpoint 2 (after completion): the coroutine is now back in a state
    // where the caller may re-check its own stop_token before the next step;
    // this method just reports the transfer's outcome.
    if (status != cudaSuccess) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_Unknown,
            std::string("GPU async copy failed: ") + cudaGetErrorString(status),
            std::source_location::current(),
        });
    }
    co_return Result<void>{};
}

auto GpuStreamQueue::StreamCount() const noexcept -> std::size_t {
    return streams_.size();
}

}  // namespace sai::runtime
