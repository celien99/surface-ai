#include <sai/image/roi.h>

#include <algorithm>
#include <cstring>
#include <source_location>

#include <sai/image/surface_image.h>

namespace sai::image {

auto ROI::BoundingBox() const noexcept -> Rect {
    if (regions.empty()) {
        return Rect{};
    }

    std::size_t min_x = regions.front().x;
    std::size_t min_y = regions.front().y;
    std::size_t max_x = regions.front().x + regions.front().width;
    std::size_t max_y = regions.front().y + regions.front().height;
    for (const Rect& region : regions) {
        min_x = std::min(min_x, region.x);
        min_y = std::min(min_y, region.y);
        max_x = std::max(max_x, region.x + region.width);
        max_y = std::max(max_y, region.y + region.height);
    }
    return Rect{min_x, min_y, max_x - min_x, max_y - min_y};
}

// 目前只裁剪第一个区域到一张新的拥有型 SurfaceImage；多区域合成（拼贴/掩膜叠加）推迟到后续任务。
auto ROI::Apply(const Image& src, const ROI& roi) -> Result<std::unique_ptr<Image>> {
    const auto dimension_mismatch = [](const char* msg) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Image_DimensionMismatch,
            msg,
            std::source_location::current(),
        });
    };

    if (roi.IsEmpty()) {
        return dimension_mismatch("ROI has no regions to apply");
    }

    const Rect region = roi.regions.front();
    const ImageMeta& meta = src.Meta();
    const bool out_of_bounds = region.width == 0 || region.height == 0 ||
                               region.x + region.width > meta.width ||
                               region.y + region.height > meta.height;
    if (out_of_bounds) {
        return dimension_mismatch("ROI region exceeds source image bounds");
    }

    const std::size_t bytes_per_pixel = BytesPerPixel(meta);
    const std::size_t src_stride = meta.width * bytes_per_pixel;
    const std::size_t dst_stride = region.width * bytes_per_pixel;
    std::vector<std::uint8_t> bytes(dst_stride * region.height);

    const std::uint8_t* src_data = src.Data();
    for (std::size_t row = 0; row < region.height; ++row) {
        const std::uint8_t* src_row =
            src_data + (region.y + row) * src_stride + region.x * bytes_per_pixel;
        std::memcpy(bytes.data() + row * dst_stride, src_row, dst_stride);
    }

    ImageMeta cropped = meta;
    cropped.width = region.width;
    cropped.height = region.height;
    return std::make_unique<SurfaceImage>(SurfaceImage::FromOwnedBuffer(std::move(bytes), cropped));
}

}  // namespace sai::image
