// detection_types_test.cpp — Task 7: DetectionResult + AnomalyMap + RegionProposal + Config 默认值
#include <gtest/gtest.h>

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>
#include <sai/detection/patch_core.h>

// ── AnomalyMap ────────────────────────────────────────────────

TEST(AnomalyMapTest, AtReturnsCorrectScore) {
    sai::detection::AnomalyMap map;
    map.grid_h = 3;
    map.grid_w = 4;
    map.scores = {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
    };
    EXPECT_FLOAT_EQ(map.At(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(map.At(0, 3), 4.0F);
    EXPECT_FLOAT_EQ(map.At(1, 1), 6.0F);
    EXPECT_FLOAT_EQ(map.At(2, 2), 11.0F);
}

TEST(AnomalyMapTest, MaxScore) {
    sai::detection::AnomalyMap map;
    map.grid_h = 2;
    map.grid_w = 3;
    map.scores = {1.0F, 3.0F, 2.0F, 0.5F, 4.5F, 2.5F};
    EXPECT_FLOAT_EQ(map.MaxScore(), 4.5F);
}

TEST(AnomalyMapTest, MaxScoreEmptyReturnsZero) {
    sai::detection::AnomalyMap map;
    EXPECT_FLOAT_EQ(map.MaxScore(), 0.0F);
}

TEST(AnomalyMapTest, IsDefective) {
    sai::detection::AnomalyMap map;
    map.grid_h = 1;
    map.grid_w = 2;
    map.scores = {0.3F, 0.8F};
    EXPECT_TRUE(map.IsDefective(0.5F));
    EXPECT_FALSE(map.IsDefective(0.9F));
    EXPECT_FALSE(map.IsDefective(0.8F));  // > threshold → false (equal is not defective)
}

// ── RegionProposal ────────────────────────────────────────────

TEST(RegionProposalTest, FieldsMatch) {
    sai::core::Rect rect{10, 20, 100, 200};
    sai::detection::RegionProposal prop{rect, 0.9F, 0.5F, 500};

    EXPECT_EQ(prop.bounding_box.x, 10U);
    EXPECT_EQ(prop.bounding_box.y, 20U);
    EXPECT_EQ(prop.bounding_box.width, 100U);
    EXPECT_EQ(prop.bounding_box.height, 200U);
    EXPECT_EQ(prop.bounding_box.Area(), 20000U);
    EXPECT_FALSE(prop.bounding_box.IsEmpty());

    EXPECT_FLOAT_EQ(prop.max_anomaly_score, 0.9F);
    EXPECT_FLOAT_EQ(prop.mean_anomaly_score, 0.5F);
    EXPECT_EQ(prop.area_pixels, 500U);
}

// ── DetectionResult ───────────────────────────────────────────

TEST(DetectionResultTest, IsDefectiveDelegatesToImageLevelScore) {
    sai::detection::DetectionResult result;
    result.image_level_score = 0.75F;

    EXPECT_TRUE(result.IsDefective(0.5F));   // 0.75 > 0.5 → defective
    EXPECT_FALSE(result.IsDefective(0.8F));  // 0.75 ≤ 0.8 → not defective
    EXPECT_FALSE(result.IsDefective(0.75F)); // 0.75 ≤ 0.75 → not defective (strict >)
}

// ── PatchCore::Config Defaults ────────────────────────────────

TEST(PatchCoreConfigTest, DefaultValuesMatchSpec) {
    sai::detection::PatchCore::Config cfg;

    EXPECT_EQ(cfg.anomaly_threshold, 0.8F);
    EXPECT_EQ(cfg.k_nearest, 1U);
    EXPECT_EQ(cfg.gaussian_sigma, 4U);
    EXPECT_EQ(cfg.image_width, 518U);
    EXPECT_EQ(cfg.image_height, 518U);
    EXPECT_EQ(cfg.patch_size, 14U);
    EXPECT_EQ(cfg.embed_dim, 1024U);
    EXPECT_TRUE(cfg.feature_bank_path.empty());
}
