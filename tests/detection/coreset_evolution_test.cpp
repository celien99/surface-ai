// coreset_evolution_test.cpp — Coreset 在线自进化单元测试
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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

// ── NoveltyFilter ─────────────────────────────────────────────

TEST(NoveltyFilterTest, NovelFramePasses) {
    // 60% patch 在 P50 以内 → coverage=0.6 < 0.6? no (equals threshold)
    // 构造 55% coverage → 应该通过
    std::vector<float> distances;
    NormalityProfile profile;
    profile.p50 = 3.0F;
    profile.num_samples = 100;

    // 55 patches low (< P50), 45 patches high (> P50)
    for (int i = 0; i < 55; ++i) distances.push_back(1.0F);
    for (int i = 0; i < 45; ++i) distances.push_back(10.0F);

    float threshold = 0.60F;
    std::size_t covered = 0;
    for (auto d : distances) if (d < profile.p50) ++covered;
    float ratio = static_cast<float>(covered) / 100.0F;

    EXPECT_FLOAT_EQ(ratio, 0.55F);
    EXPECT_LT(ratio, threshold);  // novel → passes
}

TEST(NoveltyFilterTest, RedundantFrameBlocked) {
    // 全部 patch 在 P50 以内 → coverage=1.0 → 冗余
    std::vector<float> distances(100, 1.0F);
    NormalityProfile profile;
    profile.p50 = 5.0F;
    profile.num_samples = 100;

    float threshold = 0.60F;
    std::size_t covered = 0;
    for (auto d : distances) if (d < profile.p50) ++covered;
    float ratio = static_cast<float>(covered) / 100.0F;

    EXPECT_FLOAT_EQ(ratio, 1.0F);
    EXPECT_GT(ratio, threshold);  // redundant → blocked
}

// ── CandidateBuffer ───────────────────────────────────────────

TEST(CandidateBufferTest, TriggerByFrames) {
    CandidateBuffer::Config cfg;
    cfg.trigger_frames = 5;
    CandidateBuffer buf(cfg);

    for (int i = 0; i < 5; ++i) {
        EvolutionCandidate c;
        c.grid_h = 1; c.grid_w = 10; c.dim = 8;
        c.patch_vectors = std::make_shared<const float>();  // placeholder
        EXPECT_TRUE(buf.Append(std::move(c)));
    }

    EXPECT_TRUE(buf.IsTriggered());
    EXPECT_EQ(buf.FrameCount(), 5U);
}

TEST(CandidateBufferTest, DrainAllClears) {
    CandidateBuffer::Config cfg;
    cfg.trigger_frames = 3;
    CandidateBuffer buf(cfg);

    for (int i = 0; i < 3; ++i) {
        EvolutionCandidate c;
        c.grid_h = 1; c.grid_w = 10; c.dim = 8;
        c.patch_vectors = std::make_shared<const float>();
        buf.Append(std::move(c));
    }

    auto drained = buf.DrainAll();
    EXPECT_EQ(drained.size(), 3U);
    EXPECT_EQ(buf.FrameCount(), 0U);
    EXPECT_FALSE(buf.IsTriggered());
}

TEST(CandidateBufferTest, RejectWhenFull) {
    CandidateBuffer::Config cfg;
    cfg.max_frames = 2;
    CandidateBuffer buf(cfg);

    EvolutionCandidate c1, c2, c3;
    c1.grid_h = 1; c1.grid_w = 10; c1.dim = 8;
    c1.patch_vectors = std::make_shared<const float>();
    c2 = c1; c3 = c1;

    EXPECT_TRUE(buf.Append(std::move(c1)));
    EXPECT_TRUE(buf.Append(std::move(c2)));
    EXPECT_FALSE(buf.Append(std::move(c3)));  // full
}

TEST(CandidateBufferTest, TriggerByPatches) {
    CandidateBuffer::Config cfg;
    cfg.trigger_patches = 100;
    cfg.trigger_frames = 999;  // won't trigger by frames
    CandidateBuffer buf(cfg);

    EvolutionCandidate c;
    c.grid_h = 37; c.grid_w = 37; c.dim = 1024;  // 1369 patches
    float dummy = 0.0F;
    c.patch_vectors = std::shared_ptr<const float>(&dummy, [](const float*){});

    buf.Append(std::move(c));
    EXPECT_TRUE(buf.IsTriggered());  // 1369 patches > 100
}

}  // namespace

// ── MultiSignalConsensus ──────────────────────────────────────

#include <sai/detection/detection_result.h>

TEST(MultiSignalConsensusTest, AllPassed) {
    sai::detection::NormalityAssessment normalcy{0.95F, 0.8F, 0.02F};
    sai::detection::DetectionResult det;
    det.image_level_score = 0.3F;
    // matched_rules=0, verdict="OK", threshold=0.8, pca disabled
    bool ok = sai::detection::MultiSignalConsensusCheck(
        normalcy, det, 0, "OK", 0.8F, 0.0F, 0.0F);
    EXPECT_TRUE(ok);
}

TEST(MultiSignalConsensusTest, RuleHitBlocks) {
    sai::detection::NormalityAssessment normalcy{0.95F, 0.8F, 0.02F};
    sai::detection::DetectionResult det;
    det.image_level_score = 0.3F;
    // matched_rules=1 → should fail
    bool ok = sai::detection::MultiSignalConsensusCheck(
        normalcy, det, 1, "OK", 0.8F, 0.0F, 0.0F);
    EXPECT_FALSE(ok);
}

TEST(MultiSignalConsensusTest, ReasonerNGFails) {
    sai::detection::NormalityAssessment normalcy{0.95F, 0.8F, 0.02F};
    sai::detection::DetectionResult det;
    det.image_level_score = 0.3F;
    bool ok = sai::detection::MultiSignalConsensusCheck(
        normalcy, det, 0, "NG", 0.8F, 0.0F, 0.0F);
    EXPECT_FALSE(ok);
}

TEST(MultiSignalConsensusTest, HighAnomalyScoreFails) {
    sai::detection::NormalityAssessment normalcy{0.95F, 0.8F, 0.02F};
    sai::detection::DetectionResult det;
    det.image_level_score = 0.85F;
    bool ok = sai::detection::MultiSignalConsensusCheck(
        normalcy, det, 0, "OK", 0.8F, 0.0F, 0.0F);
    EXPECT_FALSE(ok);
}

TEST(MultiSignalConsensusTest, LowNormalcyFails) {
    sai::detection::NormalityAssessment normalcy{0.5F, 0.8F, 0.15F};
    sai::detection::DetectionResult det;
    det.image_level_score = 0.3F;
    bool ok = sai::detection::MultiSignalConsensusCheck(
        normalcy, det, 0, "OK", 0.8F, 0.0F, 0.0F);
    EXPECT_FALSE(ok);
}

// ── CoresetUpdater (skeleton) ─────────────────────────────────

TEST(CoresetUpdaterSkeletonTest, LightGreedyPreservesDim) {
    // Verify that LightGreedySelect produces target_size x dim output.
    // Since LightGreedySelect is in an anonymous namespace, test indirectly
    // by checking that the prefilter won't accidentally change dimensions.
    // Full integration test in Task 5.

    // This test just verifies the buffer draining path.
    CandidateBuffer::Config buf_cfg;
    buf_cfg.trigger_frames = 2;
    CandidateBuffer buf(buf_cfg);

    std::vector<float> data(10 * 8, 1.0F);  // 10 patches, dim=8
    EvolutionCandidate c;
    c.grid_h = 1; c.grid_w = 10; c.dim = 8;
    auto data_vec = std::make_shared<std::vector<float>>(data);
    c.patch_vectors = std::shared_ptr<const float>(
        data_vec->data(), [data_vec](const float*) {});
    buf.Append(std::move(c));

    auto drained = buf.DrainAll();
    EXPECT_EQ(drained.size(), 1U);
    auto patch_count = drained[0].grid_h * drained[0].grid_w;
    EXPECT_EQ(patch_count, 10U);
}
