// surface_image.h — 预处理完毕的图像
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <sai/image/image.h>
#include <sai/memory/memory_pool.h>

namespace sai::image {

using sai::memory::IMemoryPool;
using sai::memory::PooledPtr;

class SurfaceImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& pool, ImageMeta meta) noexcept
        -> Result<SurfaceImage>;
    [[nodiscard]] static auto FromPinned(PooledPtr<std::uint8_t> pinned, ImageMeta meta) noexcept
        -> SurfaceImage;
    // 追加工厂（Task 11 记录的批准偏差）：见 raw_image.h 同名注释。
    [[nodiscard]] static auto FromOwnedBuffer(std::vector<std::uint8_t> bytes, ImageMeta meta) noexcept
        -> SurfaceImage;

    SurfaceImage(SurfaceImage&&) noexcept = default;
    auto operator=(SurfaceImage&&) noexcept -> SurfaceImage& = default;
    ~SurfaceImage() override;
    SurfaceImage(const SurfaceImage&) = delete;
    auto operator=(const SurfaceImage&) -> SurfaceImage& = delete;

    auto Release() noexcept -> void override;

private:
    // 见 raw_image.h 同名注释：持有 PooledPtr<uint8_t> 而非裸指针，析构自动归还池。
    explicit SurfaceImage(PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept;
    SurfaceImage(std::vector<std::uint8_t> owned, ImageMeta meta) noexcept;  // FromOwnedBuffer 专用

    PooledPtr<std::uint8_t> buffer_{};
    std::vector<std::uint8_t> owned_bytes_{};
};

}  // namespace sai::image
