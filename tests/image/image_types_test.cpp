#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <sai/image/image.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

#include "host_test_pool.h"

namespace {

using sai::image::ByteSize;
using sai::image::ImageMeta;
using sai::image::PixelFormat;
using sai::image::RawImage;
using sai::image::SurfaceImage;
using sai::memory::MemoryPoolConfig;
using sai::test::HostTestPool;

auto MakeMeta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = 1;
    meta.pixel_format = PixelFormat::Mono8;
    meta.frame_index = 7;
    return meta;
}

TEST(RawImageTest, FromPoolYieldsValidImageMatchingMetaAndSize) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 4096, .slab_count = 4});
    const ImageMeta meta = MakeMeta(32, 16);

    auto image = RawImage::FromPool(pool, meta);

    ASSERT_TRUE(image.has_value());
    EXPECT_TRUE(image->IsValid());
    EXPECT_NE(image->Data(), nullptr);
    EXPECT_EQ(image->SizeBytes(), ByteSize(meta));
    EXPECT_EQ(image->SizeBytes(), 32u * 16u);
    EXPECT_EQ(image->Meta().width, 32u);
    EXPECT_EQ(image->Meta().height, 16u);
    EXPECT_EQ(image->Meta().frame_index, 7u);
}

TEST(RawImageTest, MoveLeavesSourceInvalid) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 4096, .slab_count = 4});
    auto original = RawImage::FromPool(pool, MakeMeta(32, 16));
    ASSERT_TRUE(original.has_value());
    const std::uint8_t* raw_ptr = original->Data();

    RawImage moved = std::move(*original);

    EXPECT_FALSE(original->IsValid());
    EXPECT_EQ(original->Data(), nullptr);
    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(moved.Data(), raw_ptr);
}

TEST(RawImageTest, PoolBackedDestructorReturnsSlab) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 4096, .slab_count = 4});
    ASSERT_EQ(pool.AvailableSlabCount(), 4u);

    {
        auto image = RawImage::FromPool(pool, MakeMeta(32, 16));
        ASSERT_TRUE(image.has_value());
        EXPECT_EQ(pool.AvailableSlabCount(), 3u);
    }

    EXPECT_EQ(pool.AvailableSlabCount(), 4u);
}

TEST(RawImageTest, FromBufferIsNonOwningAndDoesNotTouchCallerBuffer) {
    std::vector<std::uint8_t> caller_buffer(64, 0xAB);
    const ImageMeta meta = MakeMeta(8, 8);

    {
        auto image = RawImage::FromBuffer(caller_buffer.data(), caller_buffer.size(), meta);
        EXPECT_TRUE(image.IsValid());
        EXPECT_EQ(image.Data(), caller_buffer.data());
        EXPECT_EQ(image.SizeBytes(), 64u);
    }  // dtor must NOT free / mutate the caller buffer

    EXPECT_EQ(caller_buffer.size(), 64u);
    EXPECT_EQ(caller_buffer[0], 0xAB);
    EXPECT_EQ(caller_buffer[63], 0xAB);
}

TEST(RawImageTest, FromOwnedBufferOwnsBytesAndDataStableAcrossMove) {
    std::vector<std::uint8_t> bytes(16);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(i);
    }
    const ImageMeta meta = MakeMeta(4, 4);

    RawImage image = RawImage::FromOwnedBuffer(std::move(bytes), meta);
    ASSERT_TRUE(image.IsValid());
    const std::uint8_t* data_before = image.Data();
    EXPECT_EQ(image.SizeBytes(), 16u);
    EXPECT_EQ(image.Data()[5], 5u);

    RawImage moved = std::move(image);

    EXPECT_FALSE(image.IsValid());
    EXPECT_EQ(moved.Data(), data_before);  // vector move preserves allocation
    EXPECT_EQ(moved.Data()[5], 5u);        // owned bytes survive the move
}

TEST(RawImageTest, ReleaseMakesImageInvalidAndReturnsSlab) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 4096, .slab_count = 4});
    auto image = RawImage::FromPool(pool, MakeMeta(32, 16));
    ASSERT_TRUE(image.has_value());
    ASSERT_EQ(pool.AvailableSlabCount(), 3u);

    image->Release();

    EXPECT_FALSE(image->IsValid());
    EXPECT_EQ(image->Data(), nullptr);
    EXPECT_EQ(pool.AvailableSlabCount(), 4u);  // Release returns the slab early
}

TEST(SurfaceImageTest, FromPinnedYieldsValidImage) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 4096, .slab_count = 4});
    const ImageMeta meta = MakeMeta(32, 16);
    auto pinned = pool.Acquire(ByteSize(meta));
    ASSERT_TRUE(pinned.has_value());

    SurfaceImage image = SurfaceImage::FromPinned(std::move(*pinned), meta);

    EXPECT_TRUE(image.IsValid());
    EXPECT_NE(image.Data(), nullptr);
    EXPECT_EQ(image.SizeBytes(), ByteSize(meta));
    EXPECT_EQ(pool.AvailableSlabCount(), 3u);
}

TEST(SurfaceImageTest, FromOwnedBufferOwnsBytes) {
    std::vector<std::uint8_t> bytes(9, 0x42);
    const ImageMeta meta = MakeMeta(3, 3);

    SurfaceImage image = SurfaceImage::FromOwnedBuffer(std::move(bytes), meta);

    ASSERT_TRUE(image.IsValid());
    EXPECT_EQ(image.SizeBytes(), 9u);
    EXPECT_EQ(image.Data()[0], 0x42);
    EXPECT_EQ(image.Data()[8], 0x42);
}

}  // namespace
