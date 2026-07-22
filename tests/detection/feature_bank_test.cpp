#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <sai/detection/feature_bank.h>
#include <sai/embedding/embedding.h>

namespace {

using sai::detection::FeatureBank;
using sai::embedding::Embedding;
using sai::embedding::EmbeddingMeta;
using sai::embedding::EmbeddingType;

// Build a 4-D embedding with three well-separated clusters in the first two
// dimensions. FPS (BuildWithGreedyCoreset) should pick one representative from
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

    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.type = EmbeddingType::Patch;
    meta.dim = kDim;
    meta.count = vectors.size() / kDim;
    meta.grid = {meta.count, 1};
    auto emb = Embedding::FromCpu(std::move(vectors), std::move(meta));

    std::vector<const Embedding*> ptrs{&emb};
    auto bank_result = FeatureBank::BuildWithGreedyCoreset(ptrs, kDim, 3);
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

}  // namespace
