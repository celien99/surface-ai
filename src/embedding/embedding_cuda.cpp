// embedding_cuda.cpp — Embedding GPU→CPU async download（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 Embedding::ToCpuAsync：
//   1. 校验当前数据在 GPU 上（on_gpu_ == true）
//   2. 从 PinnedPool 分配 host 端 staging buffer
//   3. cudaMemcpy DeviceToHost 将 GPU 特征数据搬移到 pinned buffer
//   4. 将 pinned buffer 数据拷贝到 cpu_data_ vector
//   5. 释放 GPU device buffer，设置 on_gpu_ = false
//
// 当前使用同步 cudaMemcpy（非真正异步）；co_return 后调用者通过
// coroutine_handle::resume() 获取结果。未来可升级为
// GpuStreamQueue::EnqueueAsyncCopy + co_await 的零拷贝异步路径。

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

    // 同步 D2H memcpy（非真正异步；未来升级为 EnqueueAsyncCopy + co_await）
    cudaError_t err = cudaMemcpy(
        pinned_buf.Get(), device_buffer_.Get(), bytes,
        cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        // pinned_buf 将在函数结束时通过 RAII 自动归还 PinnedPool
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_GpuError,
            std::string("cudaMemcpy DeviceToHost failed in ToCpuAsync: ") +
                cudaGetErrorString(err),
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
