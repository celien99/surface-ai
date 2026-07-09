#include "host_test_pool.h"

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <sai/memory/memory_pool.h>

namespace {

using sai::memory::MemoryPoolConfig;
using sai::memory::PooledPtr;
using sai::test::HostTestPool;

}  // namespace

TEST(PooledPtrTest, AcquireReturnsValidHandleWithUseCountOne) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});

    auto result = pool.Acquire(32);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->IsValid());
    EXPECT_EQ(result->UseCount(), 1);
    EXPECT_EQ(result->SizeBytes(), 64u);
}

TEST(PooledPtrTest, CopyIncrementsUseCount) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});
    auto original = pool.Acquire(32);
    ASSERT_TRUE(original.has_value());

    PooledPtr<std::uint8_t> copy = *original;

    EXPECT_EQ(original->UseCount(), 2);
    EXPECT_EQ(copy.UseCount(), 2);
    EXPECT_EQ(copy.Get(), original->Get());
}

TEST(PooledPtrTest, MoveDoesNotChangeUseCountAndInvalidatesSource) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});
    auto original = pool.Acquire(32);
    ASSERT_TRUE(original.has_value());
    std::uint8_t* raw_ptr = original->Get();

    PooledPtr<std::uint8_t> moved = std::move(*original);

    EXPECT_FALSE(original->IsValid());
    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(moved.UseCount(), 1);
    EXPECT_EQ(moved.Get(), raw_ptr);
}

TEST(PooledPtrTest, DestroyingLastCopyReturnsSlabToPool) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});
    ASSERT_EQ(pool.AvailableSlabCount(), 4u);

    {
        auto handle = pool.Acquire(32);
        ASSERT_TRUE(handle.has_value());
        EXPECT_EQ(pool.AvailableSlabCount(), 3u);

        {
            PooledPtr<std::uint8_t> copy = *handle;
            EXPECT_EQ(handle->UseCount(), 2);
            EXPECT_EQ(pool.AvailableSlabCount(), 3u);
        }  // copy destroyed here, but handle still holds a reference

        EXPECT_EQ(handle->UseCount(), 1);
        EXPECT_EQ(pool.AvailableSlabCount(), 3u);
    }  // handle destroyed here, refcount reaches zero

    EXPECT_EQ(pool.AvailableSlabCount(), 4u);
}

TEST(PooledPtrTest, RequestExceedingSlabSizeReturnsRequestExceedsSlabSize) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});

    auto result = pool.Acquire(128);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Memory_RequestExceedsSlabSize);
}

TEST(PooledPtrTest, ExhaustingAllSlabsReturnsPoolExhaustedOnNextAcquire) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 2});

    auto first = pool.Acquire(32);
    auto second = pool.Acquire(32);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    auto third = pool.Acquire(32);

    ASSERT_FALSE(third.has_value());
    EXPECT_EQ(third.error().code, sai::ErrorCode::Memory_PoolExhausted);
}

TEST(PooledPtrTest, ExplicitReleaseReturnsSlabAndClearsHandle) {
    HostTestPool pool(MemoryPoolConfig{.slab_size = 64, .slab_count = 4});
    auto handle = pool.Acquire(32);
    ASSERT_TRUE(handle.has_value());
    ASSERT_EQ(pool.AvailableSlabCount(), 3u);

    pool.Release(*handle);

    EXPECT_FALSE(handle->IsValid());
    EXPECT_EQ(pool.AvailableSlabCount(), 4u);
}
