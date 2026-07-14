#include <gtest/gtest.h>
#include <QGuiApplication>
#include <QImage>
#include <QSize>
#include <cstdint>
#include <vector>

#include "sai/visualization/frame_provider.h"
#include "sai/image/raw_image.h"
#include "sai/image/surface_image.h"
#include "sai/image/image.h"

using namespace sai::visualization;
using namespace sai::image;

// ---------------------------------------------------------------------------
// FrameProvider: requestImage after RegisterFrame returns a valid RGBA QImage
// ---------------------------------------------------------------------------
TEST(FrameProviderTest, RequestImageAfterRegisterFrameReturnsValidQImage) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    FrameProvider provider;

    // Build a 2x2 RGB8 image with known pixel colours
    std::vector<std::uint8_t> pixels = {
        255, 0,   0,   // (0,0) red
        0,   255, 0,   // (1,0) green
        0,   0,   255, // (0,1) blue
        255, 255, 0    // (1,1) yellow
    };

    ImageMeta meta;
    meta.width        = 2;
    meta.height       = 2;
    meta.channels     = 3;
    meta.pixel_format = PixelFormat::RGB8;

    auto raw = SurfaceImage::FromOwnedBuffer(std::move(pixels), meta);

    provider.RegisterFrame(1, raw);

    QSize size;
    auto result = provider.requestImage(QStringLiteral("frame?t=1"), &size, QSize());

    ASSERT_FALSE(result.isNull());
    EXPECT_EQ(result.width(), 2);
    EXPECT_EQ(result.height(), 2);
    EXPECT_EQ(result.format(), QImage::Format_RGBA8888);

    // Verify pixel (0,0) = red:  (R,G,B,A) = (255, 0, 0, 255)
    auto* bits = result.bits();
    EXPECT_EQ(bits[0], 255);  // R
    EXPECT_EQ(bits[1], 0);    // G
    EXPECT_EQ(bits[2], 0);    // B
    EXPECT_EQ(bits[3], 255);  // A

    // Verify pixel (1,0) = green
    EXPECT_EQ(bits[4 * 1 + 0], 0);
    EXPECT_EQ(bits[4 * 1 + 1], 255);
    EXPECT_EQ(bits[4 * 1 + 2], 0);
    EXPECT_EQ(bits[4 * 1 + 3], 255);
}

// ---------------------------------------------------------------------------
// RegisterRawFrame + requestImage round-trip (Mono8 path)
// ---------------------------------------------------------------------------
TEST(FrameProviderTest, RegisterRawFrameAndRequestReturnsValidQImage) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    FrameProvider provider;

    // Build a 4x4 Mono8 image filled with 50% gray
    std::vector<std::uint8_t> pixels(16, 128);  // 4*4 = 16 bytes

    ImageMeta meta;
    meta.width        = 4;
    meta.height       = 4;
    meta.channels     = 1;
    meta.pixel_format = PixelFormat::Mono8;

    auto raw = RawImage::FromOwnedBuffer(std::move(pixels), meta);

    provider.RegisterRawFrame(42, raw);

    QSize size;
    auto result = provider.requestImage(QStringLiteral("frame?t=42"), &size, QSize());

    ASSERT_FALSE(result.isNull());
    EXPECT_EQ(result.width(), 4);
    EXPECT_EQ(result.height(), 4);
    EXPECT_EQ(result.format(), QImage::Format_RGBA8888);

    // Every pixel should be (128,128,128,255)
    auto* bits = result.bits();
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(bits[i * 4 + 0], 128);
        EXPECT_EQ(bits[i * 4 + 1], 128);
        EXPECT_EQ(bits[i * 4 + 2], 128);
        EXPECT_EQ(bits[i * 4 + 3], 255);
    }
}

// ---------------------------------------------------------------------------
// Missing frame returns null QImage
// ---------------------------------------------------------------------------
TEST(FrameProviderTest, RequestImageMissingFrameReturnsEmpty) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    FrameProvider provider;

    QSize size;
    auto result = provider.requestImage(QStringLiteral("frame?t=999"), &size, QSize());

    EXPECT_TRUE(result.isNull());
}

// ---------------------------------------------------------------------------
// Invalid id string returns null QImage
// ---------------------------------------------------------------------------
TEST(FrameProviderTest, InvalidIdReturnsNull) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    FrameProvider provider;
    QSize size;

    EXPECT_TRUE(provider.requestImage(QStringLiteral("badformat"), &size, QSize()).isNull());
    EXPECT_TRUE(provider.requestImage(QStringLiteral("frame?t=abc"), &size, QSize()).isNull());
    EXPECT_TRUE(provider.requestImage(QStringLiteral(""), &size, QSize()).isNull());
}

// ---------------------------------------------------------------------------
// Ring-buffer overwrite: frame 1 and frame 17 share slot 1%16 = 17%16 = 1
// ---------------------------------------------------------------------------
TEST(FrameProviderTest, RingBufferOverwriteReplacesSlot) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    FrameProvider provider;

    // Register frame 1
    {
        std::vector<std::uint8_t> px1(2 * 2 * 3, 255);  // white 2x2 RGB8
        ImageMeta meta;
        meta.width        = 2;
        meta.height       = 2;
        meta.channels     = 3;
        meta.pixel_format = PixelFormat::RGB8;
        auto img = SurfaceImage::FromOwnedBuffer(std::move(px1), meta);
        provider.RegisterFrame(1, img);
    }

    // Overwrite same slot with frame 17
    {
        std::vector<std::uint8_t> px17(2 * 2 * 3, 0);  // black 2x2 RGB8
        ImageMeta meta;
        meta.width        = 2;
        meta.height       = 2;
        meta.channels     = 3;
        meta.pixel_format = PixelFormat::RGB8;
        auto img = SurfaceImage::FromOwnedBuffer(std::move(px17), meta);
        provider.RegisterFrame(17, img);
    }

    QSize size;

    // Frame 1 was overwritten → should return null
    EXPECT_TRUE(provider.requestImage(QStringLiteral("frame?t=1"), &size, QSize()).isNull());

    // Frame 17 should be available
    auto result = provider.requestImage(QStringLiteral("frame?t=17"), &size, QSize());
    ASSERT_FALSE(result.isNull());
    // Black pixel check
    EXPECT_EQ(result.bits()[0], 0);
    EXPECT_EQ(result.bits()[1], 0);
    EXPECT_EQ(result.bits()[2], 0);
}
