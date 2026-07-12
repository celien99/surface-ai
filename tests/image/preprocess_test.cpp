#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/error.h>
#include <sai/image/preprocess.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

namespace {

using sai::ErrorCode;
using sai::ErrorInfo;
using sai::Result;
using sai::image::Compose;
using sai::image::Image;
using sai::image::ImageMeta;
using sai::image::MakeDebayer;
using sai::image::MakeFlatField;
using sai::image::PixelFormat;
using sai::image::RawImage;

auto MakeMono8Meta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = 1;
    meta.pixel_format = PixelFormat::Mono8;
    return meta;
}

// 以拥有型堆缓冲构造 Mono8 图像，返回 unique_ptr<Image>（预处理链的通用消费类型）。
auto MakeMono8(std::size_t width, std::size_t height, std::uint8_t fill = 0)
    -> std::unique_ptr<Image> {
    std::vector<std::uint8_t> bytes(width * height, fill);
    return std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeMono8Meta(width, height)));
}

// ---- Compose ----

TEST(Preprocess, ComposeShortCircuitsOnError) {
    int ran2 = 0;
    auto step1 = [](std::unique_ptr<Image>) -> Result<std::unique_ptr<Image>> {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Image_PreprocessFailed, "boom", {}});
    };
    auto step2 = [&](std::unique_ptr<Image> i) -> Result<std::unique_ptr<Image>> {
        ++ran2;
        return i;
    };
    auto chain = Compose({step1, step2});
    auto out = chain(MakeMono8(2, 2));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_PreprocessFailed);
    EXPECT_EQ(ran2, 0);
}

TEST(Preprocess, ComposeRunsStepsInOrder) {
    std::vector<int> order;
    auto step1 = [&](std::unique_ptr<Image> i) -> Result<std::unique_ptr<Image>> {
        order.push_back(1);
        return i;
    };
    auto step2 = [&](std::unique_ptr<Image> i) -> Result<std::unique_ptr<Image>> {
        order.push_back(2);
        return i;
    };
    auto chain = Compose({step1, step2});
    auto out = chain(MakeMono8(2, 2));
    ASSERT_TRUE(out.has_value());
    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
}

TEST(Preprocess, ComposeEmptyReturnsInputUnchanged) {
    auto chain = Compose({});
    auto in = MakeMono8(3, 3, 7);
    const Image* raw = in.get();
    auto out = chain(std::move(in));
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(out->get(), raw);  // same object passed straight through
}

// ---- MakeFlatField ----

TEST(Preprocess, FlatFieldUniformCorrectionLeavesInputUnchanged) {
    // 校正帧全部 == 参考灰度 128：ref_mean == 128，逐像素 out = in * 128 / 128 == in。
    RawImage corr =
        RawImage::FromOwnedBuffer(std::vector<std::uint8_t>(64, 128), MakeMono8Meta(8, 8));
    auto fn = MakeFlatField(corr);

    auto in = MakeMono8(8, 8, 100);
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    for (std::size_t i = 0; i < img.SizeBytes(); ++i) {
        EXPECT_EQ(img.Data()[i], 100u) << "index " << i;
    }
}

TEST(Preprocess, FlatFieldDarkCornerScalesPixelUp) {
    // 校正帧大部分 128，仅角点 (0,0) 偏暗 == 64：该像素增益 ~ ref_mean/64 ≈ 2x。
    std::vector<std::uint8_t> corr_bytes(64, 128);
    corr_bytes[0] = 64;
    RawImage corr = RawImage::FromOwnedBuffer(std::move(corr_bytes), MakeMono8Meta(8, 8));
    auto fn = MakeFlatField(corr);

    auto in = MakeMono8(8, 8, 100);
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    // ref_mean = (63*128 + 64)/64 = 127. corner: 100*127/64 = 198 (clamped from below 255).
    EXPECT_GT(img.Data()[0], 100u);
    EXPECT_EQ(img.Data()[0], 198u);
    // a normal pixel (corr==128): 100*127/128 = 99, essentially unchanged.
    EXPECT_EQ(img.Data()[1], 99u);
}

TEST(Preprocess, FlatFieldRejectsDimensionMismatch) {
    RawImage corr =
        RawImage::FromOwnedBuffer(std::vector<std::uint8_t>(16, 128), MakeMono8Meta(4, 4));
    auto fn = MakeFlatField(corr);
    auto out = fn(MakeMono8(8, 8, 100));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_DimensionMismatch);
}

// ---- MakeDebayer ----

TEST(Preprocess, DebayerBayerRG8ToRgb8) {
    // 4x4 BayerRG8, RGGB。第一个 2x2 quad: R=10 G=20 / G=30 B=40。
    std::vector<std::uint8_t> bayer = {
        10, 20, 11, 21,  //
        30, 40, 31, 41,  //
        12, 22, 13, 23,  //
        32, 42, 33, 43,  //
    };
    ImageMeta meta;
    meta.width = 4;
    meta.height = 4;
    meta.channels = 1;
    meta.pixel_format = PixelFormat::BayerRG8;
    auto in = std::make_unique<RawImage>(RawImage::FromOwnedBuffer(std::move(bayer), meta));

    auto fn = MakeDebayer();
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Meta().channels, 3u);
    EXPECT_EQ(img.Meta().pixel_format, PixelFormat::RGB8);
    EXPECT_EQ(img.Meta().width, 4u);
    EXPECT_EQ(img.Meta().height, 4u);
    EXPECT_EQ(img.SizeBytes(), 4u * 4u * 3u);
    // pixel (0,0) takes quad (0,0): R=10, G=20 (top-right green), B=40.
    EXPECT_EQ(img.Data()[0], 10u);
    EXPECT_EQ(img.Data()[1], 20u);
    EXPECT_EQ(img.Data()[2], 40u);
}

TEST(Preprocess, DebayerRejectsNonBayerFormat) {
    auto fn = MakeDebayer();
    auto out = fn(MakeMono8(4, 4));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_UnsupportedPixelFormat);
}

}  // namespace
