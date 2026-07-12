#include <sai/image/surface_image.h>

#include <utility>

namespace sai::image {

SurfaceImage::SurfaceImage(PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept
    : Image(buffer.Get(), ByteSize(meta), meta), buffer_(std::move(buffer)) {}

SurfaceImage::SurfaceImage(std::vector<std::uint8_t> owned, ImageMeta meta) noexcept
    : Image(owned.data(), owned.size(), meta), owned_bytes_(std::move(owned)) {}

SurfaceImage::~SurfaceImage() = default;

auto SurfaceImage::FromPool(IMemoryPool& pool, ImageMeta meta) noexcept -> Result<SurfaceImage> {
    return pool.Acquire(ByteSize(meta)).map([&meta](PooledPtr<std::uint8_t> buffer) {
        return SurfaceImage(std::move(buffer), meta);
    });
}

auto SurfaceImage::FromPinned(PooledPtr<std::uint8_t> pinned, ImageMeta meta) noexcept
    -> SurfaceImage {
    return SurfaceImage(std::move(pinned), meta);
}

auto SurfaceImage::FromOwnedBuffer(std::vector<std::uint8_t> bytes, ImageMeta meta) noexcept
    -> SurfaceImage {
    return SurfaceImage(std::move(bytes), meta);
}

auto SurfaceImage::Release() noexcept -> void {
    buffer_ = PooledPtr<std::uint8_t>{};  // 移动赋空句柄：若持有 slab 则归还池
    owned_bytes_.clear();
    owned_bytes_.shrink_to_fit();
    Image::Release();
}

}  // namespace sai::image
