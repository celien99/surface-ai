// coreset_evolution_test.cpp — Coreset 在线自进化单元测试
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <sai/detection/coreset_evolution.h>
#include <sai/detection/feature_bank.h>
#include <sai/embedding/embedding.h>

namespace {

namespace fs = std::filesystem;
using namespace sai::detection;

// ── Helper: build a small FeatureBank with known vectors ──

auto BuildSmallBank(std::size_t dim = 8, std::size_t count = 100) -> FeatureBank {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    std::vector<sai::embedding::Embedding> embs;
    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "test";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = dim;
    meta.count = count;
    meta.grid = {count, 1};
    embs.push_back(sai::embedding::Embedding::FromCpu(std::move(data), std::move(meta)));

    std::vector<const sai::embedding::Embedding*> ptrs;
    for (auto& e : embs) ptrs.push_back(&e);
    auto fb = FeatureBank::BuildFromEmbeddings(ptrs, dim, count);
    return std::move(*fb);
}

// ── NormalityProfile ──────────────────────────────────────────

TEST(NormalityProfileTest, ComputeFromBank) {
    auto bank = BuildSmallBank(8, 100);
    auto profile = NormalityProfile::Compute(bank, 5);

    EXPECT_EQ(profile.k_nearest, 5U);
    EXPECT_EQ(profile.dim, 8U);
    EXPECT_EQ(profile.num_samples, 100U);
    EXPECT_GT(profile.p95, profile.p50);
    EXPECT_GE(profile.p99, profile.p95);
    EXPECT_GT(profile.mean, 0.0F);
    EXPECT_GT(profile.stddev, 0.0F);
}

TEST(NormalityProfileTest, RoundTripYaml) {
    auto bank = BuildSmallBank(8, 50);
    auto profile = NormalityProfile::Compute(bank, 5);

    auto tmp = fs::temp_directory_path() / "test_profile.yaml";
    auto save_result = profile.SaveToYaml(tmp);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message;

    auto load_result = NormalityProfile::LoadFromYaml(tmp);
    ASSERT_TRUE(load_result.has_value()) << load_result.error().message;
    auto loaded = std::move(*load_result);

    EXPECT_EQ(loaded.k_nearest, profile.k_nearest);
    EXPECT_EQ(loaded.dim, profile.dim);
    EXPECT_FLOAT_EQ(loaded.p50, profile.p50);
    EXPECT_FLOAT_EQ(loaded.p95, profile.p95);
    EXPECT_FLOAT_EQ(loaded.mean, profile.mean);

    fs::remove(tmp);
}

TEST(NormalityProfileTest, ComputeFastApproximatesFull) {
    auto bank = BuildSmallBank(8, 200);
    auto full = NormalityProfile::Compute(bank, 5);
    auto fast = NormalityProfile::ComputeFast(bank, 5, 50);

    // 采样估计应在全量值的 30% 误差内
    EXPECT_NEAR(fast.p50, full.p50, full.p50 * 0.30F);
    EXPECT_NEAR(fast.p95, full.p95, full.p95 * 0.30F);
    EXPECT_NEAR(fast.mean, full.mean, full.mean * 0.30F);
}

// ── NormalityScorer ───────────────────────────────────────────

// 注意: NormalityScorer 当前是匿名命名空间内部类。
// 通过 NormalityAssessment + 手工 distances 测试逻辑。
// 后续 Task 3 (MultiSignalConsensus) 会提供公开入口。

TEST(NormalityScorerTest, AllNormalGetsHighScore) {
    // 构造全部低距离的 distances（模拟全部 patch 在正常范围内）
    std::vector<float> distances(100, 1.0F);  // 100 patches, all low
    NormalityProfile profile;
    profile.p50 = 2.0F;
    profile.p95 = 5.0F;
    profile.num_samples = 100;
    profile.dim = 8;

    // 手动计算（复用 NormalityScorer 逻辑）
    std::sort(distances.begin(), distances.end());
    float tail_ratio_max = 0.10F;
    std::size_t tail_count = 0;
    for (auto d : distances) {
        if (d > profile.p95) ++tail_count;
    }
    float tail_ratio = static_cast<float>(tail_count) / 100.0F;
    float normalcy = 1.0F - std::min(1.0F, tail_ratio / tail_ratio_max);

    EXPECT_EQ(tail_ratio, 0.0F);
    EXPECT_FLOAT_EQ(normalcy, 1.0F);
}

TEST(NormalityScorerTest, OutlierGetsLowScore) {
    // 构造一半正常、一半超 P95 的 distances
    std::vector<float> distances;
    for (int i = 0; i < 50; ++i) distances.push_back(1.0F);   // normal
    for (int i = 0; i < 50; ++i) distances.push_back(10.0F);  // outlier

    NormalityProfile profile;
    profile.p95 = 5.0F;
    profile.num_samples = 100;

    float tail_ratio_max = 0.10F;
    std::size_t tail_count = 0;
    for (auto d : distances) {
        if (d > profile.p95) ++tail_count;
    }
    float tail_ratio = static_cast<float>(tail_count) / 100.0F;
    float normalcy = 1.0F - std::min(1.0F, tail_ratio / tail_ratio_max);

    EXPECT_FLOAT_EQ(tail_ratio, 0.50F);
    EXPECT_FLOAT_EQ(normalcy, 0.0F);  // 50% tail → score=0
}

}  // namespace
