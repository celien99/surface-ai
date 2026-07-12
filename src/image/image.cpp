#include <sai/image/image.h>

namespace sai::image {

Image::Image(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept
    : data_(data), size_bytes_(size_bytes), meta_(meta) {}

Image::Image(Image&& other) noexcept
    : Resource(std::move(other)),
      data_(other.data_),
      size_bytes_(other.size_bytes_),
      meta_(other.meta_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
}

auto Image::operator=(Image&& other) noexcept -> Image& {
    if (this == &other) {
        return *this;
    }
    Resource::operator=(std::move(other));
    data_ = other.data_;
    size_bytes_ = other.size_bytes_;
    meta_ = other.meta_;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    return *this;
}

auto Image::Release() noexcept -> void {
    data_ = nullptr;
    size_bytes_ = 0;
}

}  // namespace sai::image
