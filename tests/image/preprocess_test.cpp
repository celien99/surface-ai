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
using sai::image::CalibrationParams;
using sai::image::Compose;
using sai::image::Image;
using sai::image::ImageMeta;
using sai::image::MakeCalibration;
using sai::image::MakeDebayer;
using sai::image::MakeFlatField;
using sai::image::MakeHDR;
using sai::image::MakeResize;
using sai::image::MakeWhiteBalance;
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

auto MakeRGB8Meta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = 3;
    meta.pixel_format = PixelFormat::RGB8;
    return meta;
}

auto MakeBGR8Meta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = 3;
    meta.pixel_format = PixelFormat::BGR8;
    return meta;
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

// ---- MakeWhiteBalance ----

TEST(Preprocess, WhiteBalanceRgb8AppliesPerChannelGain) {
    auto fn = MakeWhiteBalance(1.2f, 1.0f, 0.8f);
    std::vector<std::uint8_t> bytes = {100, 100, 100};
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeRGB8Meta(1, 1)));
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Data()[0], 120u);  // R: 100*1.2=120
    EXPECT_EQ(img.Data()[1], 100u);  // G: 100*1.0=100
    EXPECT_EQ(img.Data()[2], 80u);   // B: 100*0.8=80
}

TEST(Preprocess, WhiteBalanceClampsTo255) {
    auto fn = MakeWhiteBalance(2.0f, 0.5f, 2.0f);
    std::vector<std::uint8_t> bytes = {200, 100, 200};
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeRGB8Meta(1, 1)));
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Data()[0], 255u);  // R: 200*2.0=400 clamped to 255
    EXPECT_EQ(img.Data()[1], 50u);   // G: 100*0.5=50
    EXPECT_EQ(img.Data()[2], 255u);  // B: 200*2.0=400 clamped to 255
}

TEST(Preprocess, WhiteBalanceRejectsNonRgb) {
    auto fn = MakeWhiteBalance(1.0f, 1.0f, 1.0f);
    auto out = fn(MakeMono8(4, 4));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_UnsupportedPixelFormat);
}

TEST(Preprocess, WhiteBalanceBgr8AppliesGainsInBgrOrder) {
    auto fn = MakeWhiteBalance(1.0f, 2.0f, 3.0f);
    // BGR8 single pixel: channel order is B, G, R
    // r_gain=1.0, g_gain=2.0, b_gain=3.0 → applied as B*b_gain, G*g_gain, R*r_gain
    std::vector<std::uint8_t> bytes = {10, 20, 30};  // B=10, G=20, R=30
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeBGR8Meta(1, 1)));
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Data()[0], 30u);   // B: 10*3.0=30
    EXPECT_EQ(img.Data()[1], 40u);   // G: 20*2.0=40
    EXPECT_EQ(img.Data()[2], 30u);   // R: 30*1.0=30
}

// ---- MakeResize ----

TEST(Preprocess, ResizeDownscale4x4Rgb8To2x2) {
    // 4x4 RGB8: top-left 2x2 block has known values for bilinear verification.
    // output(0,0) maps to source(0.5, 0.5) → blends (0,0),(1,0),(0,1),(1,1) equally.
    std::vector<std::uint8_t> bytes(4 * 4 * 3, 0);
    auto setPx = [&](std::size_t x, std::size_t y, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
        std::size_t off = (y * 4 + x) * 3;
        bytes[off + 0] = r;
        bytes[off + 1] = g;
        bytes[off + 2] = b;
    };
    setPx(0, 0, 100, 0, 0);
    setPx(1, 0, 0, 100, 0);
    setPx(0, 1, 0, 0, 100);
    setPx(1, 1, 100, 100, 100);

    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeRGB8Meta(4, 4)));

    auto fn = MakeResize(2, 2);
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Meta().width, 2u);
    EXPECT_EQ(img.Meta().height, 2u);
    EXPECT_EQ(img.Meta().channels, 3u);
    EXPECT_EQ(img.Meta().pixel_format, PixelFormat::RGB8);
    EXPECT_EQ(img.SizeBytes(), 12u);
    // output(0,0): bilinear equal-weight blend → (50, 50, 50)
    EXPECT_EQ(img.Data()[0], 50u);
    EXPECT_EQ(img.Data()[1], 50u);
    EXPECT_EQ(img.Data()[2], 50u);
}

TEST(Preprocess, ResizeZeroTargetWidthReturnsError) {
    auto fn = MakeResize(0, 100);
    auto out = fn(MakeMono8(4, 4));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_PreprocessFailed);
}

TEST(Preprocess, ResizeZeroTargetHeightReturnsError) {
    auto fn = MakeResize(100, 0);
    auto out = fn(MakeMono8(4, 4));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_PreprocessFailed);
}

TEST(Preprocess, ResizeSameSizePreservesPixels) {
    // 2x2 Mono8, same-size resize → identity
    std::vector<std::uint8_t> bytes = {10, 20, 30, 40};
    ImageMeta meta = MakeMono8Meta(2, 2);
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), meta));

    auto fn = MakeResize(2, 2);
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Meta().width, 2u);
    EXPECT_EQ(img.Meta().height, 2u);
    EXPECT_EQ(img.SizeBytes(), 4u);
    EXPECT_EQ(img.Data()[0], 10u);
    EXPECT_EQ(img.Data()[1], 20u);
    EXPECT_EQ(img.Data()[2], 30u);
    EXPECT_EQ(img.Data()[3], 40u);
}

// ---- MakeCalibration ----

TEST(Preprocess, CalibrationIdentityMatrixReturnsCopy) {
    CalibrationParams params;
    params.camera_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    params.dist_coeffs = {0, 0, 0, 0, 0};
    params.pixel_scale_mm = 2.5;

    // 2x2 RGB8 with distinct pixels
    std::vector<std::uint8_t> bytes = {
        10, 20, 30,  50, 60, 70,
        80, 90, 100, 110, 120, 130,
    };
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), MakeRGB8Meta(2, 2)));

    auto fn = MakeCalibration(params);
    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Meta().width, 2u);
    EXPECT_EQ(img.Meta().height, 2u);
    // Pixel (0,0) maps to exact source integer → identity copy
    EXPECT_EQ(img.Data()[0], 10u);
    EXPECT_EQ(img.Data()[1], 20u);
    EXPECT_EQ(img.Data()[2], 30u);
    EXPECT_EQ(img.Data()[3], 50u);
    EXPECT_EQ(img.Data()[4], 60u);
    EXPECT_EQ(img.Data()[5], 70u);
    EXPECT_EQ(img.Data()[6], 80u);
    EXPECT_EQ(img.Data()[7], 90u);
    EXPECT_EQ(img.Data()[8], 100u);
    EXPECT_EQ(img.Data()[9], 110u);
    EXPECT_EQ(img.Data()[10], 120u);
    EXPECT_EQ(img.Data()[11], 130u);
}

// ---- MakeHDR ----

TEST(Preprocess, HdrSingleFrameStretch) {
    auto fn = MakeHDR(1);
    // Low contrast Mono8: min=50, max=200
    std::vector<std::uint8_t> bytes = {50, 125, 200};
    ImageMeta meta = MakeMono8Meta(3, 1);
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), meta));

    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    // 50 → 0, 200 → 255, 125 → (125-50)*255/(200-50)=75*255/150=127.5→127
    EXPECT_EQ(img.Data()[0], 0u);
    EXPECT_EQ(img.Data()[1], 127u);
    EXPECT_EQ(img.Data()[2], 255u);
}

TEST(Preprocess, HdrZeroExposuresReturnsError) {
    auto fn = MakeHDR(0);
    auto out = fn(MakeMono8(4, 4));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_PreprocessFailed);
}

TEST(Preprocess, HdrUniformImageUnchanged) {
    auto fn = MakeHDR(1);
    // All pixels equal → min==max, no stretch
    std::vector<std::uint8_t> bytes = {100, 100, 100};
    ImageMeta meta = MakeMono8Meta(3, 1);
    auto in = std::make_unique<RawImage>(
        RawImage::FromOwnedBuffer(std::move(bytes), meta));

    auto out = fn(std::move(in));
    ASSERT_TRUE(out.has_value());
    const Image& img = **out;
    EXPECT_EQ(img.Data()[0], 100u);
    EXPECT_EQ(img.Data()[1], 100u);
    EXPECT_EQ(img.Data()[2], 100u);
}

}  // namespace
