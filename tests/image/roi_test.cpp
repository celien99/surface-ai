#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <sai/image/raw_image.h>
#include <sai/image/roi.h>

namespace {

using sai::image::ImageMeta;
using sai::image::PixelFormat;
using sai::image::RawImage;
using sai::image::Rect;
using sai::image::ROI;

auto MakeMono8Meta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = 1;
    meta.pixel_format = PixelFormat::Mono8;
    return meta;
}

TEST(RoiTest, IsEmptyReflectsRegions) {
    EXPECT_TRUE(ROI{}.IsEmpty());
    ROI roi{.regions = {Rect{0, 0, 2, 2}}};
    EXPECT_FALSE(roi.IsEmpty());
}

TEST(RoiTest, BoundingBoxSpansTwoRects) {
    ROI roi{.regions = {Rect{1, 2, 3, 4}, Rect{5, 1, 2, 2}}};

    const Rect box = roi.BoundingBox();

    // min-x = 1, min-y = 1, max-x = max(1+3, 5+2) = 7, max-y = max(2+4, 1+2) = 6
    EXPECT_EQ(box.x, 1u);
    EXPECT_EQ(box.y, 1u);
    EXPECT_EQ(box.width, 6u);
    EXPECT_EQ(box.height, 5u);
}

TEST(RoiTest, BoundingBoxOfEmptyIsDefaultRect) {
    const Rect box = ROI{}.BoundingBox();

    EXPECT_EQ(box.x, 0u);
    EXPECT_EQ(box.y, 0u);
    EXPECT_EQ(box.width, 0u);
    EXPECT_EQ(box.height, 0u);
}

TEST(RoiTest, ApplyCropsFirstRegionWithCorrectPixels) {
    // 4x4 Mono8, value = row*10 + col
    std::vector<std::uint8_t> pixels(16);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            pixels[row * 4 + col] = static_cast<std::uint8_t>(row * 10 + col);
        }
    }
    RawImage src = RawImage::FromBuffer(pixels.data(), pixels.size(), MakeMono8Meta(4, 4));
    // crop the 2x2 sub-rect at (1,1): rows 1-2, cols 1-2
    ROI roi{.regions = {Rect{1, 1, 2, 2}}};

    auto cropped = ROI::Apply(src, roi);

    ASSERT_TRUE(cropped.has_value());
    const sai::image::Image& out = **cropped;
    ASSERT_NE(out.Data(), nullptr);
    EXPECT_EQ(out.Meta().width, 2u);
    EXPECT_EQ(out.Meta().height, 2u);
    EXPECT_EQ(out.SizeBytes(), 4u);
    EXPECT_EQ(out.Data()[0], 11u);  // (row 1, col 1)
    EXPECT_EQ(out.Data()[1], 12u);  // (row 1, col 2)
    EXPECT_EQ(out.Data()[2], 21u);  // (row 2, col 1)
    EXPECT_EQ(out.Data()[3], 22u);  // (row 2, col 2)
}

TEST(RoiTest, ApplyRejectsOversizedRegion) {
    std::vector<std::uint8_t> pixels(16, 0);
    RawImage src = RawImage::FromBuffer(pixels.data(), pixels.size(), MakeMono8Meta(4, 4));
    ROI roi{.regions = {Rect{2, 2, 4, 4}}};  // extends beyond 4x4

    auto cropped = ROI::Apply(src, roi);

    ASSERT_FALSE(cropped.has_value());
    EXPECT_EQ(cropped.error().code, sai::ErrorCode::Image_DimensionMismatch);
}

}  // namespace
