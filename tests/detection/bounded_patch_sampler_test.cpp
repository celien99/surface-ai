#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <sai/detection/bounded_patch_sampler.h>

namespace {

using sai::detection::BoundedPatchSampler;

TEST(BoundedPatchSamplerTest, FixedInputIsDeterministicAcrossChunks) {
    constexpr std::size_t kDim = 2;
    constexpr std::size_t kCapacity = 4;
    std::vector<float> patches;
    for (std::size_t i = 0; i < 20; ++i) {
        patches.push_back(static_cast<float>(i));
        patches.push_back(static_cast<float>(i + 100));
    }

    BoundedPatchSampler whole(kDim, kCapacity);
    whole.Add(patches.data(), 20);

    BoundedPatchSampler chunked(kDim, kCapacity);
    chunked.Add(patches.data(), 7);
    chunked.Add(patches.data() + 7 * kDim, 13);

    EXPECT_EQ(whole.Size(), kCapacity);
    EXPECT_EQ(whole.Vectors(), chunked.Vectors());
}

TEST(BoundedPatchSamplerTest, StorageNeverExceedsCapacity) {
    constexpr std::size_t kDim = 3;
    std::vector<float> patches(1000 * kDim, 1.0F);
    BoundedPatchSampler sampler(kDim, 5);

    sampler.Add(patches.data(), 1000);

    EXPECT_EQ(sampler.Size(), 5U);
    EXPECT_EQ(sampler.Vectors().size(), 5U * kDim);
}

}  // namespace
