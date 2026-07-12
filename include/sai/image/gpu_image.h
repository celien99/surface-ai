// gpu_image.h — GPU 显存驻留图像 + GPU 搬运步骤声明（批次 2.2）
#pragma once

#include <sai/image/image.h>
#include <sai/image/preprocess.h>
#include <sai/memory/memory_pool.h>

namespace sai::image {

using sai::memory::IMemoryPool;
using sai::memory::PooledPtr;

// GpuImage：显存驻留图像，设备内存由 GpuPool（IMemoryPool）分配。
// Data() 返回设备内存指针——宿主代码解引用是 UB，只能传给 CUDA API。
// 构造仅通过 FromPool(GpuPool&) 工厂，移动不拷贝，析构归还 slab。
class GpuImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& gpu_pool, ImageMeta meta) noexcept
        -> Result<GpuImage>;

    GpuImage(GpuImage&&) noexcept = default;
    auto operator=(GpuImage&&) noexcept -> GpuImage& = default;
    ~GpuImage() override;
    GpuImage(const GpuImage&) = delete;
    auto operator=(const GpuImage&) -> GpuImage& = delete;

    [[nodiscard]] auto IsGpuImage() const noexcept -> bool override { return true; }
    auto Release() noexcept -> void override;

private:
    explicit GpuImage(PooledPtr<std::uint8_t> device_buffer, ImageMeta meta) noexcept;
    PooledPtr<std::uint8_t> buffer_{};
};

// GPU 搬运步骤工厂：返回一个 PreprocessFn，执行 HtoD 拷贝 → [GPU 操作占位] → DtoH 拷贝，
// 并在此过程中显式填充/排空中转缓冲（修复遗留偏差 D1，见 spec §4.3）。
//
// pinned_pool 和 gpu_queue 必须在该 PreprocessFn 的整个生命周期内保持存活。
// 此声明可移植 include（无 CUDA 头文件），函数体在 gpu_preprocess.cpp（CUDA 门控）中。
//
// 输入要求：SurfaceImage（CPU 侧预处理完毕的 RGB8/Mono8 图像）。
// 输出：一新 SurfaceImage（从 DtoH 的 pinned buffer 构造），所有权转移给调用方。
namespace sai::memory { class PinnedPool; }
namespace sai::runtime { class GpuStreamQueue; }

[[nodiscard]] auto MakeGpuUploadStep(sai::memory::PinnedPool& pinned_pool,
                                     sai::runtime::GpuStreamQueue& gpu_queue) -> PreprocessFn;

}  // namespace sai::image
