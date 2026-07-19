// embedding_cuda.cpp — Embedding GPU→CPU async download（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 Embedding::ToCpuAsync：
//   1. 校验当前数据在 GPU 上（on_gpu_ == true）
//   2. 从 PinnedPool 分配 host 端 staging buffer
//   3. cudaMemcpyAsync DeviceToHost 将 GPU 特征数据搬移到 pinned buffer
//      （使用独立 CUDA stream，避免默认流隐式串行化）
//   4. 将 pinned buffer 数据拷贝到 cpu_data_ vector
//   5. 释放 GPU device buffer，设置 on_gpu_ = false
//
// Batch T3: 将同步 cudaMemcpy 替换为 cudaMemcpyAsync + 独立 stream，
// 消除默认 CUDA stream 的隐式全局同步，允许多线程 GPU 操作并发执行。

#include <sai/embedding/embedding.h>

#include <cstring>
#include <source_location>

#include <cuda_runtime.h>

#include <sai/runtime/gpu_stream_queue.h>
#include <sai/memory/memory_pool.h>
#include <sai/memory/pinned_pool.h>

namespace sai::embedding {

auto Embedding::ToCpuAsync(sai::runtime::GpuStreamQueue& /*queue*/,
                            sai::memory::PinnedPool& pinned,
                            std::stop_token /*token*/) noexcept
    -> sai::runtime::Task<sai::Result<void>> {
    // 已在 CPU 上 — 无需操作
    if (!on_gpu_) {
        co_return Result<void>{};
    }

    const std::size_t bytes = SizeBytes();
    if (bytes == 0) {
        co_return Result<void>{};
    }

    // 从 PinnedPool 分配 host-side staging buffer
    auto pinned_buf_result = pinned.Acquire(bytes);
    if (!pinned_buf_result) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Memory_AllocationFailed,
            "PinnedPool::Acquire failed in ToCpuAsync — "
            "not enough slabs for embedding of " +
                std::to_string(bytes) + " bytes",
            std::source_location::current(),
        });
    }
    auto pinned_buf = std::move(*pinned_buf_result);

    // Async D2H memcpy on an independent CUDA stream — avoids default-stream
    // implicit serialization so that other GPU ops (inference, D2D copies on
    // other streams) can execute concurrently with this transfer.
    cudaStream_t d2h_stream = nullptr;
    cudaError_t stream_err = cudaStreamCreate(&d2h_stream);
    if (stream_err != cudaSuccess) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_GpuError,
            std::string("cudaStreamCreate failed in ToCpuAsync: ") +
                cudaGetErrorString(stream_err),
            std::source_location::current(),
        });
    }

    cudaError_t err = cudaMemcpyAsync(
        pinned_buf.Get(), device_buffer_.Get(), bytes,
        cudaMemcpyDeviceToHost, d2h_stream);

    if (err != cudaSuccess) {
        cudaStreamDestroy(d2h_stream);
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_GpuError,
            std::string("cudaMemcpyAsync DeviceToHost failed in ToCpuAsync: ") +
                cudaGetErrorString(err),
            std::source_location::current(),
        });
    }

    cudaError_t sync_err = cudaStreamSynchronize(d2h_stream);
    cudaStreamDestroy(d2h_stream);

    if (sync_err != cudaSuccess) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_GpuError,
            std::string("cudaStreamSynchronize failed in ToCpuAsync: ") +
                cudaGetErrorString(sync_err),
            std::source_location::current(),
        });
    }

    // 从 pinned buffer 拷贝到 cpu_data_ vector（零拷贝 pinned 仅对 GPU 可见；
    // CPU 端仍然需要一次 memcpy 进入 vector 堆内存）
    const auto* src = static_cast<const float*>(
        static_cast<const void*>(pinned_buf.Get()));
    const std::size_t count = meta_.count * meta_.dim;
    cpu_data_.assign(src, src + count);

    // 释放 GPU device buffer — 数据已安全拷贝到 CPU
    // PooledPtr 没有 Release() 方法；赋空触发 move-assignment，旧值通过 RAII 归还
    device_buffer_ = sai::memory::PooledPtr<std::uint8_t>{};
    on_gpu_ = false;

    // pinned_buf 在作用域结束时通过 RAII 自动归还 PinnedPool

    co_return Result<void>{};
}

}  // namespace sai::embedding
