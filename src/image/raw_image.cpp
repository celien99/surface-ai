#include <sai/image/raw_image.h>

#include <utility>

namespace sai::image {

RawImage::RawImage(PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept
    : Image(buffer.Get(), ByteSize(meta), meta), buffer_(std::move(buffer)) {}

RawImage::RawImage(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept
    : Image(data, size_bytes, meta) {}

RawImage::RawImage(std::vector<std::uint8_t> owned, ImageMeta meta) noexcept
    : Image(owned.data(), owned.size(), meta), owned_bytes_(std::move(owned)) {}

RawImage::~RawImage() = default;

auto RawImage::FromPool(IMemoryPool& pool, ImageMeta meta) noexcept -> Result<RawImage> {
    return pool.Acquire(ByteSize(meta)).map([&meta](PooledPtr<std::uint8_t> buffer) {
        return RawImage(std::move(buffer), meta);
    });
}

auto RawImage::FromBuffer(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept
    -> RawImage {
    return RawImage(data, size_bytes, meta);
}

auto RawImage::FromOwnedBuffer(std::vector<std::uint8_t> bytes, ImageMeta meta) noexcept
    -> RawImage {
    return RawImage(std::move(bytes), meta);
}

auto RawImage::Release() noexcept -> void {
    buffer_ = PooledPtr<std::uint8_t>{};  // 移动赋空句柄：若持有 slab 则归还池
    owned_bytes_.clear();
    owned_bytes_.shrink_to_fit();
    Image::Release();
}

}  // namespace sai::image
