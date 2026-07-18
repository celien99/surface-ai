// gpu_preprocess.cpp — GPU 搬运步骤实现（CUDA 门控，修复遗留偏差 D1）
//
// 实现 MakeGpuUploadStep：PreprocessFn 将 SurfaceImage 通过 HtoD 搬上 GPU，
// 执行 GPU 操作占位（M3 插入点），再通过 DtoH 搬回 SurfaceImage。
// 在此过程中显式填充/排空中转缓冲——这是 D1 修复的核心：
//   - HtoD 前：std::memcpy(cpu_data → pinned buffer) —— POPULATE
//   - DtoH 后：SurfaceImage::FromPinned(pinned buffer) —— DRAIN
// GpuStreamQueue::EnqueueAsyncCopy 的接口不修改——填充/排空完全是调用方的职责。
//
// 本文件仅在 find_package(CUDAToolkit) 成功时编译——见 src/image/CMakeLists.txt 的门控逻辑。

#include <sai/image/gpu_image.h>

#include <coroutine>
#include <cstring>
#include <memory>
#include <source_location>
#include <stop_token>
#include <utility>

#include <cuda_runtime.h>

#include <sai/core/error.h>
#include <sai/memory/pinned_pool.h>
#include <sai/runtime/gpu_stream_queue.h>
#include <sai/runtime/task.h>
#include <sai/image/surface_image.h>

namespace sai::image {

using sai::memory::PinnedPool;
using sai::runtime::CopyDirection;
using sai::runtime::GpuStreamQueue;
using sai::runtime::Task;

namespace {

// SyncAwaitable<T>：将 Task<T>（std::coroutine_handle<TaskPromise<T>>）适配为可 co_await 的类型。
// await_suspend 中通过手动的 while(!done()) resume() 循环将内部协程驱动至完成，
// 然后返回 void（使外层协程保持挂起，控制权交还给驱动循环）。
// 驱动循环再次 resume() 外层协程时，编译器调用 await_resume() 提取 Result<T>。
template <typename T>
struct SyncAwaitable {
    Task<T> handle_;

    [[nodiscard]] bool await_ready() noexcept {
        return handle_.done();
    }

    void await_suspend(std::coroutine_handle<> /*caller*/) noexcept {
        while (!handle_.done()) {
            handle_.resume();
        }
    }

    [[nodiscard]] auto await_resume() noexcept -> Result<T> {
        auto r = handle_.promise().GetResult();
        handle_.destroy();
        return r;
    }
};

// GpuUploadCoroutine：内部协程，实现 HtoD → [GPU 占位] → DtoH 的完整搬运管线。
// 返回 Task<Result<std::unique_ptr<Image>>>，由 MakeGpuUploadStep 的 PreprocessFn
// lambda 通过 while(!done()) resume() 循环同步驱动至完成。
//
// D1 修复（标注在对应行）：
//   - HtoD：调用方 std::memcpy 填充中转缓冲（POPULATE）
//   - DtoH：调用方从中转缓冲构造 SurfaceImage（DRAIN）
// GpuStreamQueue::EnqueueAsyncCopy 的接口未修改。
auto GpuUploadCoroutine(PinnedPool& pinned_pool, GpuStreamQueue& gpu_queue,
                        std::unique_ptr<Image> image)
    -> Task<std::unique_ptr<Image>> {
    auto meta = image->Meta();
    auto size = image->SizeBytes();
    const std::uint8_t* const cpu_data = image->Data();
    std::stop_token token;  // PreprocessFn 单帧同步执行，无需外部取消

    // ================================================================
    // HtoD 路径
    // ================================================================
    auto transit = pinned_pool.Acquire(size);
    if (!transit) {
        co_return tl::make_unexpected(transit.error());
    }

    // D1 POPULATE：将 CPU 侧像素数据复制到 pinned 中转缓冲中。
    // 这是 D1 修复的前半部分——EnqueueAsyncCopy 不做填充，调用方负责。
    std::memcpy(transit->Get(), cpu_data, size);

    // 提交异步 HtoD 拷贝。PooledPtr 移动进 EnqueueAsyncCopy 的协程帧：
    // 拷贝完成后帧析构归还 slab——我们不再需要这个 pinned buffer。
    {
        auto htod_result = co_await SyncAwaitable<void>{
            gpu_queue.EnqueueAsyncCopy(std::move(*transit), size,
                                       CopyDirection::HostToDevice, token)};
        if (!htod_result) {
            co_return tl::make_unexpected(htod_result.error());
        }
    }

    // ================================================================
    // GPU 操作占位符（里程碑 3 AI 推理核心将在此处插入）
    // co_await gpu_inference(gpu_image, ...);
    // ================================================================

    // ================================================================
    // DtoH 路径
    // ================================================================
    auto transit2 = pinned_pool.Acquire(size);
    if (!transit2) {
        co_return tl::make_unexpected(transit2.error());
    }

    // 提交异步 DtoH 拷贝。注意：此处传递 *transit2（拷贝）而非 std::move(*transit2),
    // 因为我们必须在拷贝完成后仍持有 PooledPtr 以读取 GPU 写入的数据。
    // EnqueueAsyncCopy 的协程帧持有拷贝（refcount→2），帧析构后 refcount→1，
    // 调用方仍持有有效句柄。
    {
        auto dtoh_result = co_await SyncAwaitable<void>{
            gpu_queue.EnqueueAsyncCopy(*transit2, size,
                                       CopyDirection::DeviceToHost, token)};
        if (!dtoh_result) {
            co_return tl::make_unexpected(dtoh_result.error());
        }
    }

    // D1 DRAIN：将传输完成后的 pinned 缓冲显式转换为 SurfaceImage。
    // 这是 D1 修复的后半部分——EnqueueAsyncCopy 不做排空，调用方负责从
    // 中转缓冲中提取数据。PooledPtr 移动进 SurfaceImage 内部，析构时归还池。
    auto out = SurfaceImage::FromPinned(std::move(*transit2), meta);
    co_return std::make_unique<SurfaceImage>(std::move(out));
}

}  // namespace

auto MakeGpuUploadStep(PinnedPool& pinned_pool, GpuStreamQueue& gpu_queue) -> PreprocessFn {
    return [&pinned_pool, &gpu_queue](std::unique_ptr<Image> image)
               -> Result<std::unique_ptr<Image>> {
        if (!image || !image->IsValid()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_PreprocessFailed,
                "MakeGpuUploadStep: input image is null or invalid",
                std::source_location::current(),
            });
        }

        // 创建内部协程并同步驱动至完成。
        // GpuUploadCoroutine 使用 co_await SyncAwaitable<void>{...} 挂起，
        // SyncAwaitable::await_suspend 返回 void → 控制权回到此循环，
        // 再次 resume() 触发 await_resume() → 协程继续执行。
        auto task = GpuUploadCoroutine(pinned_pool, gpu_queue, std::move(image));
        while (!task.done()) {
            task.resume();
        }
        auto result = task.promise().GetResult();
        task.destroy();
        return result;
    };
}

}  // namespace sai::image
