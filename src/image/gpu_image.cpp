// gpu_image.cpp — GpuImage 实现（CUDA 门控，仅在 CUDAToolkit_FOUND 时编译）
//
// 本文件包装真实的 CUDA Runtime API 调用（cudaMalloc/cudaFree），用于目标平台
// （Ubuntu x64 + NVIDIA GPU）。在没有 CUDA Toolkit 的宿主机上，此文件不在源文件
// 列表中，不会被编译——见 src/image/CMakeLists.txt 的门控逻辑。

#include <sai/image/gpu_image.h>

#include <cuda_runtime.h>

#include <sai/memory/gpu_pool.h>
#include <sai/image/surface_image.h>

namespace sai::image {

GpuImage::GpuImage(PooledPtr<std::uint8_t> device_buffer, ImageMeta meta) noexcept
    : Image(device_buffer.Get(), ByteSize(meta), meta), buffer_(std::move(device_buffer)) {}

GpuImage::~GpuImage() = default;

auto GpuImage::FromPool(IMemoryPool& gpu_pool, ImageMeta meta) noexcept -> Result<GpuImage> {
    // GpuPool 是 IMemoryPool 的实现，Acquire 返回 PooledPtr<std::uint8_t>，
    // 底层 slab 指针来自 cudaMalloc——宿主代码解引用它是 UB，但传给 CUDA API 是合法的。
    return gpu_pool.Acquire(ByteSize(meta)).map([&meta](PooledPtr<std::uint8_t> buffer) {
        return GpuImage(std::move(buffer), meta);
    });
}

auto GpuImage::Release() noexcept -> void {
    buffer_ = PooledPtr<std::uint8_t>{};  // 移动赋空句柄：析构归还设备内存 slab 给 GpuPool
    Image::Release();
}

}  // namespace sai::image
