#include <sai/memory/arena_allocator.h>

#include <cstddef>

#include <gtest/gtest.h>

namespace {

struct PodLike {
    int a;
    double b;
};

}  // namespace

TEST(ArenaAllocatorTest, ConstructReturnsWorkingPointer) {
    sai::memory::ArenaAllocator arena(1024);

    auto result = arena.Construct<PodLike>(PodLike{.a = 7, .b = 3.5});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->a, 7);
    EXPECT_EQ((*result)->b, 3.5);
}

TEST(ArenaAllocatorTest, CapacityExhaustionReturnsArenaExhausted) {
    sai::memory::ArenaAllocator arena(sizeof(PodLike));

    auto first = arena.Construct<PodLike>(PodLike{.a = 1, .b = 1.0});
    ASSERT_TRUE(first.has_value());

    auto second = arena.Construct<PodLike>(PodLike{.a = 2, .b = 2.0});

    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, sai::ErrorCode::Memory_ArenaExhausted);
}

TEST(ArenaAllocatorTest, CapacityAndUsedBytesReportAcrossMultipleConstructs) {
    sai::memory::ArenaAllocator arena(1024);

    EXPECT_EQ(arena.CapacityBytes(), 1024u);
    EXPECT_EQ(arena.UsedBytes(), 0u);

    ASSERT_TRUE(arena.Construct<PodLike>(PodLike{.a = 1, .b = 1.0}).has_value());
    const std::size_t used_after_first = arena.UsedBytes();
    EXPECT_GE(used_after_first, sizeof(PodLike));
    EXPECT_EQ(arena.CapacityBytes(), 1024u);

    ASSERT_TRUE(arena.Construct<PodLike>(PodLike{.a = 2, .b = 2.0}).has_value());
    const std::size_t used_after_second = arena.UsedBytes();
    EXPECT_GT(used_after_second, used_after_first);
    EXPECT_EQ(arena.CapacityBytes(), 1024u);
}
