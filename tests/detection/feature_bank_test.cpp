#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <sai/detection/feature_bank.h>

namespace {

using sai::detection::FeatureBank;
// Build a 4-D embedding with three well-separated clusters in the first two
// dimensions. Exact FPS should pick one representative from
// each cluster when max_samples == 3, proving coverage-aware selection.
TEST(FeatureBankGreedyCoresetTest, PicksOnePerCluster) {
    constexpr std::size_t kDim = 4;
    constexpr std::size_t kPerCluster = 50;
    constexpr float kNoise = 0.5F;
    constexpr float kSeparation = 100.0F;
    constexpr float kClusterRadius = 5.0F;

    std::vector<float> vectors;
    std::vector<float> centers = {
        0.0F, 0.0F,
        kSeparation, 0.0F,
        0.0F, kSeparation,
    };

    // Deterministic "noise": alternate small offsets so points stay near center.
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < kPerCluster; ++i) {
            float sign = (i % 2 == 0) ? 1.0F : -1.0F;
            float offset = kNoise * static_cast<float>(i % 5) * sign;
            vectors.push_back(centers[c * 2 + 0] + offset);   // x
            vectors.push_back(centers[c * 2 + 1] - offset);   // y
            vectors.push_back(0.0F);                          // z
            vectors.push_back(0.0F);                          // w
        }
    }

    auto bank_result = FeatureBank::BuildGreedyFromVectors(vectors, kDim, 3);
    ASSERT_TRUE(bank_result.has_value()) << bank_result.error().message;
    auto bank = std::move(*bank_result);

    EXPECT_EQ(bank.NumSamples(), 3U);
    EXPECT_EQ(bank.Dim(), kDim);

    auto selected = bank.ExtractAllVectors();
    ASSERT_EQ(selected.size(), 3U * kDim);

    // Each selected point must be close to a distinct cluster center.
    std::vector<bool> cluster_used(3, false);
    for (std::size_t i = 0; i < 3; ++i) {
        float x = selected[i * kDim + 0];
        float y = selected[i * kDim + 1];
        for (std::size_t c = 0; c < 3; ++c) {
            float dx = x - centers[c * 2 + 0];
            float dy = y - centers[c * 2 + 1];
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < kClusterRadius) {
                cluster_used[c] = true;
            }
        }
    }
    EXPECT_TRUE(cluster_used[0]) << "No selected point near cluster 0";
    EXPECT_TRUE(cluster_used[1]) << "No selected point near cluster 1";
    EXPECT_TRUE(cluster_used[2]) << "No selected point near cluster 2";
}

TEST(FeatureBankGreedyCoresetTest, PreservesExactGreedyOrderAndTieBreak) {
    // Starting at 0: furthest is 10 (index 3), then -10 (index 4), then the
    // tied points 5/-5 have equal coverage and the smaller index 1 wins.
    const std::vector<float> vectors = {0.0F, 5.0F, -5.0F, 10.0F, -10.0F};
    auto bank = FeatureBank::BuildGreedyFromVectors(vectors, 1, 4);
    ASSERT_TRUE(bank.has_value()) << bank.error().message;
    EXPECT_EQ(bank->ExtractAllVectors(),
              (std::vector<float>{0.0F, 10.0F, -10.0F, 5.0F}));
}

TEST(FeatureBankGreedyCoresetTest, StopsWhenOnlyDuplicateVectorsRemain) {
    const std::vector<float> vectors = {
        1.0F, 2.0F,
        1.0F, 2.0F,
        1.0F, 2.0F,
    };
    auto bank = FeatureBank::BuildGreedyFromVectors(vectors, 2, 3);
    ASSERT_TRUE(bank.has_value()) << bank.error().message;
    EXPECT_EQ(bank->NumSamples(), 1U);
}

TEST(FeatureBankGreedyCoresetTest, RejectsNonFiniteVectors) {
    const std::vector<float> vectors = {0.0F, std::numeric_limits<float>::quiet_NaN()};
    auto bank = FeatureBank::BuildGreedyFromVectors(vectors, 1, 2);
    ASSERT_FALSE(bank.has_value());
}

}  // namespace
