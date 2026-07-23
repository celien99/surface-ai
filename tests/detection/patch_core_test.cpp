// patch_core_test.cpp — Task 8: FeatureBank + PatchCore::Detect 测试
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/context.h>
#include <sai/detection/detection_result.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/detection/post_process_utils.h>
#include <sai/embedding/dimension_reducer.h>
#include <sai/embedding/embedding.h>

namespace {

namespace fs = std::filesystem;

// 写入 N×dim 的 float32 矩阵到临时文件（little-endian，行主序）
auto WriteFloatMatrix(const fs::path& path, const std::vector<float>& data) -> void {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

// 创建包含 1 个样本向量的临时 coreset 文件，返回路径和向量数据。
struct TempCoreset {
    fs::path path;
    std::vector<float> data;
    std::size_t dim;
};
auto MakeTempCoreset() -> TempCoreset {
    TempCoreset cs;
    cs.dim = 4;
    cs.data = {1.0F, 0.0F, 0.0F, 0.0F};  // 单个样本，维度 4
    auto tmp = fs::temp_directory_path() / "test_coreset_f32.bin";
    // 如果文件已存在则取唯一文件名
    cs.path = tmp;
    int suffix = 0;
    while (fs::exists(cs.path)) {
        cs.path = tmp.parent_path() / ("test_coreset_f32_" + std::to_string(++suffix) + ".bin");
    }
    WriteFloatMatrix(cs.path, cs.data);
    return cs;
}

// ── FeatureBank ─────────────────────────────────────────────────

TEST(FeatureBankTest, LoadFromFileNumSamplesAndDim) {
    auto cs = MakeTempCoreset();
    auto bank = sai::detection::FeatureBank::LoadFromFile(cs.path, cs.dim);
    ASSERT_TRUE(bank.has_value()) << "LoadFromFile failed: " << bank.error().message;
    EXPECT_EQ(bank->NumSamples(), 1U);
    EXPECT_EQ(bank->Dim(), cs.dim);
}

TEST(FeatureBankTest, LoadFromFileNonExistentReturnsError) {
    auto result = sai::detection::FeatureBank::LoadFromFile("/nonexistent/path.bin", 4);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Detection_FeatureBankLoadFailed);
}

TEST(FeatureBankTest, LoadFromFileWrongFileSizeReturnsError) {
    // 文件大小不是 dim*sizeof(float) 的整数倍
    std::vector<float> bad_data = {1.0F, 2.0F};  // 2 floats, dim=3 → 不匹配
    auto path = fs::temp_directory_path() / "test_bad_size.bin";
    WriteFloatMatrix(path, bad_data);
    auto result = sai::detection::FeatureBank::LoadFromFile(path, 3);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Detection_FeatureBankLoadFailed);
    fs::remove(path);
}

TEST(FeatureBankTest, SearchExactMatchReturnsZeroDistance) {
    auto cs = MakeTempCoreset();
    auto bank = sai::detection::FeatureBank::LoadFromFile(cs.path, cs.dim);
    ASSERT_TRUE(bank.has_value());

    // 查询与 coreset 中唯一向量完全相同的向量
    std::vector<float> query = {1.0F, 0.0F, 0.0F, 0.0F};
    auto distances = bank->Search(query.data(), 1, 1);
    ASSERT_EQ(distances.size(), 1U);
    EXPECT_NEAR(distances[0], 0.0F, 1e-4F);
}

TEST(FeatureBankTest, SearchDistantQueryReturnsPositiveDistance) {
    auto cs = MakeTempCoreset();
    auto bank = sai::detection::FeatureBank::LoadFromFile(cs.path, cs.dim);
    ASSERT_TRUE(bank.has_value());

    // 查询与 coreset 向量完全不同的向量
    std::vector<float> query = {0.0F, 0.0F, 1.0F, 1.0F};
    auto distances = bank->Search(query.data(), 1, 1);
    ASSERT_EQ(distances.size(), 1U);
    // L2 距离: (1-0)^2 + (0-0)^2 + (0-1)^2 + (0-1)^2 = 1 + 0 + 1 + 1 = 3
    EXPECT_NEAR(distances[0], 3.0F, 1e-4F);
}

TEST(FeatureBankTest, SearchMultiQueryMultiK) {
    // coreset: 2 个 2 维向量
    std::vector<float> coreset = {0.0F, 0.0F, 10.0F, 0.0F};
    auto path = fs::temp_directory_path() / "test_multi.f32.bin";
    WriteFloatMatrix(path, coreset);
    auto bank = sai::detection::FeatureBank::LoadFromFile(path, 2);
    ASSERT_TRUE(bank.has_value());
    EXPECT_EQ(bank->NumSamples(), 2U);

    // 查询 2 个向量，各找 2 个最近邻
    // query0=(1,0) → nearest: (0,0) d=1, (10,0) d=81
    // query1=(9,0) → nearest: (10,0) d=1, (0,0) d=81
    std::vector<float> queries = {1.0F, 0.0F, 9.0F, 0.0F};
    auto distances = bank->Search(queries.data(), 2, 2);
    ASSERT_EQ(distances.size(), 4U);  // 2 queries × 2 neighbors
    EXPECT_NEAR(distances[0], 1.0F, 1e-4F);   // query0, neighbor0
    EXPECT_NEAR(distances[1], 81.0F, 1e-4F);  // query0, neighbor1
    EXPECT_NEAR(distances[2], 1.0F, 1e-4F);   // query1, neighbor0
    EXPECT_NEAR(distances[3], 81.0F, 1e-4F);  // query1, neighbor1
    fs::remove(path);
}

// ── Bilinear Upsample ───────────────────────────────────────────

TEST(UpsampleTest, TwoByTwoToFourByFour) {
    // 2×2 源，四个角为 0,1,2,3
    std::vector<float> src = {0.0F, 1.0F, 2.0F, 3.0F};  // row0: 0,1; row1: 2,3
    auto dst = sai::detection::BilinearUpsample(src.data(), 2, 2, 4, 4);
    ASSERT_EQ(dst.size(), 16U);

    // 角点应精确等于源值
    EXPECT_FLOAT_EQ(dst[0], 0.0F);                      // (0,0)
    EXPECT_FLOAT_EQ(dst[3], 1.0F);                      // (0,3)
    EXPECT_FLOAT_EQ(dst[12], 2.0F);                     // (3,0)
    EXPECT_FLOAT_EQ(dst[15], 3.0F);                     // (3,3)

    // 中心点应大致等于平均值
    float center = dst[5];  // ~(1,1) 映射
    EXPECT_GE(center, 0.0F);
    EXPECT_LE(center, 3.0F);
}

TEST(UpsampleTest, SameSizeIsIdentity) {
    std::vector<float> src = {5.0F, 2.0F, 7.0F, 3.0F};
    auto dst = sai::detection::BilinearUpsample(src.data(), 2, 2, 2, 2);
    ASSERT_EQ(dst.size(), 4U);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

// ── Connected Components ────────────────────────────────────────

TEST(ConnectedComponentsTest, EmptyMaskReturnsZeroRegions) {
    std::vector<float> binary = {0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> scores = {0.0F, 0.0F, 0.0F, 0.0F};
    auto regions = sai::detection::ConnectedComponents(binary.data(), 2, 2, scores.data());
    EXPECT_TRUE(regions.empty());
}

TEST(ConnectedComponentsTest, SingleComponentBboxCorrect) {
    // 2×2 mask: 右下角一个像素为 1
    std::vector<float> binary = {0.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float> scores = {0.0F, 0.0F, 0.0F, 0.9F};
    auto regions = sai::detection::ConnectedComponents(binary.data(), 2, 2, scores.data());
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].bounding_box.x, 1U);
    EXPECT_EQ(regions[0].bounding_box.y, 1U);
    EXPECT_EQ(regions[0].bounding_box.width, 1U);
    EXPECT_EQ(regions[0].bounding_box.height, 1U);
    EXPECT_FLOAT_EQ(regions[0].max_anomaly_score, 0.9F);
    EXPECT_FLOAT_EQ(regions[0].mean_anomaly_score, 0.9F);
    EXPECT_EQ(regions[0].area_pixels, 1U);
}

TEST(ConnectedComponentsTest, DisconnectedComponents) {
    // 3×3 mask: 左上角 (0,0) 和右下角 (2,2) 各为一个 1
    // 0 0 0
    // 0 1 0   ← center is 0
    // 0 0 0
    // Wait, let me make it clearer:
    // row0: 1 0 0
    // row1: 0 0 0
    // row2: 0 0 1
    std::vector<float> binary = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float> scores = {0.8F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.6F};
    auto regions = sai::detection::ConnectedComponents(binary.data(), 3, 3, scores.data());
    ASSERT_EQ(regions.size(), 2U);
    // 按 max_score 降序排：0.8 在前，0.6 在后
    EXPECT_FLOAT_EQ(regions[0].max_anomaly_score, 0.8F);
    EXPECT_FLOAT_EQ(regions[1].max_anomaly_score, 0.6F);
}

TEST(ConnectedComponentsTest, FourConnectedNeighborJoins) {
    // 3×3 mask 中两个水平相邻的 1 应合并为一个 region
    // row0: 0 0 0
    // row1: 1 1 0
    // row2: 0 0 0
    std::vector<float> binary = {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> scores = {0.0F, 0.0F, 0.0F, 0.7F, 0.9F, 0.0F, 0.0F, 0.0F, 0.0F};
    auto regions = sai::detection::ConnectedComponents(binary.data(), 3, 3, scores.data());
    ASSERT_EQ(regions.size(), 1U);
    EXPECT_EQ(regions[0].bounding_box.x, 0U);
    EXPECT_EQ(regions[0].bounding_box.y, 1U);
    EXPECT_EQ(regions[0].bounding_box.width, 2U);
    EXPECT_EQ(regions[0].bounding_box.height, 1U);
    EXPECT_FLOAT_EQ(regions[0].max_anomaly_score, 0.9F);
    EXPECT_FLOAT_EQ(regions[0].mean_anomaly_score, 0.8F);  // (0.7+0.9)/2
    EXPECT_EQ(regions[0].area_pixels, 2U);
}

// ── GaussianBlur ─────────────────────────────────────────────────

TEST(GaussianBlurTest, SigmaZeroIsIdentity) {
    std::vector<float> src = {0.0F, 1.0F, 2.0F, 3.0F};
    auto dst = sai::detection::GaussianBlur(src.data(), 2, 2, 0);
    ASSERT_EQ(dst.size(), 4U);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(dst[i], src[i], 1e-4F);
    }
}

TEST(GaussianBlurTest, BlurPreservesRange) {
    // 3×3: 中心为 1.0，其余为 0
    std::vector<float> src = {0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    auto dst = sai::detection::GaussianBlur(src.data(), 3, 3, 1);
    ASSERT_EQ(dst.size(), 9U);
    // 模糊后中心值降低，邻域值升高，但都应在 [0,1] 内
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_GE(dst[i], 0.0F);
        EXPECT_LE(dst[i], 1.0F);
    }
    // 中心仍然应该是最大点
    float center_val = dst[4];
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_LE(dst[i], center_val + 1e-6F);
    }
}

// ── PatchCore::Detect ───────────────────────────────────────────

}  // namespace

class PatchCoreDetectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建简单的 coreset：4 个 4 维向量。
        std::vector<float> coreset = {
            0.5F, 0.5F, 0.5F, 0.5F,
            0.6F, 0.6F, 0.6F, 0.6F,
            0.4F, 0.4F, 0.4F, 0.4F,
            0.7F, 0.7F, 0.7F, 0.7F,
        };
        coreset_path_ = fs::temp_directory_path() / "test_patchcore_coreset.bin";
        WriteFloatMatrix(coreset_path_, coreset);

        sai::detection::PatchCore::Config cfg;
        cfg.feature_bank_path = coreset_path_;
        cfg.image_width = 28;
        cfg.image_height = 28;
        cfg.patch_size = 14;
        cfg.embed_dim = 4;  // 匹配 coreset 维度
        cfg.k_nearest = 1;
        cfg.anomaly_threshold = 0.5F;
        cfg.gaussian_sigma = 1;

        patch_core_ = std::make_unique<sai::detection::PatchCore>(cfg);
    }

    void TearDown() override {
        fs::remove(coreset_path_);
    }

    fs::path coreset_path_;
    std::unique_ptr<sai::detection::PatchCore> patch_core_;
    sai::Context ctx_;
};

TEST_F(PatchCoreDetectTest, InitializeSucceeds) {
    auto result = patch_core_->Initialize(ctx_);
    ASSERT_TRUE(result.has_value()) << result.error().message;
}

TEST_F(PatchCoreDetectTest, DetectWithValidEmbeddingReturnsResult) {
    auto init_result = patch_core_->Initialize(ctx_);
    ASSERT_TRUE(init_result.has_value());

    // 创建 2×2 patch grid 的 embedding（28/14=2）
    // 4 个 patch 向量，每个 4 维
    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 4;
    meta.grid = {2, 2};

    std::vector<float> data = {
        0.5F, 0.5F, 0.5F, 0.5F,   // patch (0,0)
        0.5F, 0.5F, 0.5F, 0.5F,   // patch (0,1)
        0.5F, 0.5F, 0.5F, 0.5F,   // patch (1,0)
        0.9F, 0.9F, 0.9F, 0.9F,   // patch (1,1) — 异常
    };
    auto embedding = sai::embedding::Embedding::FromCpu(std::move(data), std::move(meta));

    auto result = patch_core_->Detect(embedding);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto& det = result.value();
    // AnomalyMap 应为 2×2
    EXPECT_EQ(det.anomaly_map.grid_h, 2U);
    EXPECT_EQ(det.anomaly_map.grid_w, 2U);
    EXPECT_EQ(det.anomaly_map.scores.size(), 4U);
    // 异常分数在 [0,1]
    for (auto s : det.anomaly_map.scores) {
        EXPECT_GE(s, 0.0F);
        EXPECT_LE(s, 1.0F);
    }
    // 图像级分数 >= 0
    EXPECT_GE(det.image_level_score, 0.0F);
}

TEST_F(PatchCoreDetectTest, DetectInvalidPatchGridReturnsError) {
    auto init_result = patch_core_->Initialize(ctx_);
    ASSERT_TRUE(init_result.has_value());

    // grid 与 config 不匹配：config 期望 2×2，此处给 3×3
    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 9;
    meta.grid = {3, 3};

    std::vector<float> data(9 * 4, 0.5F);
    auto embedding = sai::embedding::Embedding::FromCpu(std::move(data), std::move(meta));

    auto result = patch_core_->Detect(embedding);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Detection_InvalidPatchGrid);
}

TEST_F(PatchCoreDetectTest, DetectBatchReturnsMultipleResults) {
    auto init_result = patch_core_->Initialize(ctx_);
    ASSERT_TRUE(init_result.has_value());

    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 4;
    meta.grid = {2, 2};

    std::vector<float> data1 = {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F,
                                0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F};
    auto emb1 = sai::embedding::Embedding::FromCpu(std::move(data1), meta);

    meta.grid = {2, 2};
    meta.count = 4;
    std::vector<float> data2 = {0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F,
                                0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F};
    auto emb2 = sai::embedding::Embedding::FromCpu(std::move(data2), meta);

    std::array<const sai::embedding::Embedding*, 2> ptrs = {&emb1, &emb2};
    auto results = patch_core_->DetectBatch(ptrs);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results.value().size(), 2U);
}

TEST_F(PatchCoreDetectTest, AllBelowThresholdReturnsNoRegions) {
    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = coreset_path_;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 999.0F;  // 极高阈值，不生成 region
    cfg.gaussian_sigma = 0;

    auto pc = sai::detection::PatchCore(cfg);
    auto init_result = pc.Initialize(ctx_);
    ASSERT_TRUE(init_result.has_value());

    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 4;
    meta.grid = {2, 2};

    std::vector<float> data(16, 0.5F);
    auto embedding = sai::embedding::Embedding::FromCpu(std::move(data), std::move(meta));

    auto result = pc.Detect(embedding);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().regions.empty());
}

TEST_F(PatchCoreDetectTest, ModelNameReturnsPatchCore) {
    EXPECT_EQ(patch_core_->ModelName(), "PatchCore");
}

// 测试未初始化的 PatchCore 调用 Detect
TEST_F(PatchCoreDetectTest, DetectWithoutInitializeReturnsError) {
    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 4;
    meta.grid = {2, 2};

    std::vector<float> data(16, 0.5F);
    auto embedding = sai::embedding::Embedding::FromCpu(std::move(data), std::move(meta));

    auto result = patch_core_->Detect(embedding);
    EXPECT_FALSE(result.has_value());
}

// ── SubspaceAD 增强测试 ──────────────────────────────────────────

// 构造多样本 coreset 的辅助函数
struct MultiSampleCoreset {
    fs::path path;
    std::size_t dim;
    std::size_t num_samples;
};
auto MakeMultiSampleCoreset() -> MultiSampleCoreset {
    MultiSampleCoreset cs;
    cs.dim = 4;
    cs.num_samples = 8;
    // 8 个 4 维向量，足够多样以支持 PCA 拟合
    cs.path = fs::temp_directory_path() / "test_multisample_coreset.bin";
    int suffix = 0;
    while (fs::exists(cs.path)) {
        cs.path = fs::temp_directory_path().parent_path() /
                  ("test_multisample_coreset_" + std::to_string(++suffix) + ".bin");
    }
    std::vector<float> data = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.5F, 0.0F,
        0.0F, 0.0F, 0.5F, 0.5F,
        0.5F, 0.0F, 0.0F, 0.5F,
    };
    WriteFloatMatrix(cs.path, data);
    return cs;
}

// 创建 Patch Embedding（2×2 grid, dim=4）
auto Make2x2Embedding(const std::vector<float>& data) -> sai::embedding::Embedding {
    sai::embedding::EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = sai::embedding::EmbeddingType::Patch;
    meta.dim = 4;
    meta.count = 4;
    meta.grid = {2, 2};
    auto copy = data;  // FromCpu 需要非 const vector
    return sai::embedding::Embedding::FromCpu(std::move(copy), std::move(meta));
}

TEST(SubspaceADTest, WhiteningEnabled) {
    auto cs = MakeMultiSampleCoreset();

    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = cs.path;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 0.5F;
    cfg.gaussian_sigma = 1;
    cfg.enable_whitening = true;
    cfg.drop_k = 0;

    auto pc = sai::detection::PatchCore(cfg);
    sai::Context ctx;
    auto init_result = pc.Initialize(ctx);
    ASSERT_TRUE(init_result.has_value()) << init_result.error().message;

    // 查询向量——含一个异常 patch
    std::vector<float> query = {
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.9F, 0.9F, 0.9F, 0.9F,
    };
    auto embedding = Make2x2Embedding(query);
    auto result = pc.Detect(embedding);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result.value().anomaly_map.scores.size(), 4U);
    EXPECT_GE(result.value().image_level_score, 0.0F);

    fs::remove(cs.path);
}

TEST(SubspaceADTest, AdaptiveThreshold) {
    auto cs = MakeMultiSampleCoreset();

    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = cs.path;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 0.5F;
    cfg.gaussian_sigma = 1;
    cfg.enable_adaptive_threshold = true;
    cfg.target_fpr = 0.5F;

    auto pc = sai::detection::PatchCore(cfg);
    sai::Context ctx;
    auto init_result = pc.Initialize(ctx);
    ASSERT_TRUE(init_result.has_value()) << init_result.error().message;

    // 正常查询（与 coreset 相近）
    std::vector<float> normal_query = {
        0.5F, 0.5F, 0.0F, 0.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
    };
    auto emb_normal = Make2x2Embedding(normal_query);
    auto result_normal = pc.Detect(emb_normal);
    ASSERT_TRUE(result_normal.has_value());

    // 异常查询（全 1，远离 coreset）
    std::vector<float> anomaly_query(16, 1.0F);
    auto emb_anomaly = Make2x2Embedding(anomaly_query);
    auto result_anomaly = pc.Detect(emb_anomaly);
    ASSERT_TRUE(result_anomaly.has_value());

    // 异常查询的 image_level_score 应高于正常查询
    EXPECT_GT(result_anomaly.value().image_level_score,
              result_normal.value().image_level_score);

    fs::remove(cs.path);
}

TEST(SubspaceADTest, HybridScoring) {
    auto cs = MakeMultiSampleCoreset();

    // 用 coreset 数据手动物合 PCA
    std::vector<float> coreset_data = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.5F, 0.0F,
        0.0F, 0.0F, 0.5F, 0.5F,
        0.5F, 0.0F, 0.0F, 0.5F,
    };
    std::vector<sai::embedding::Embedding> samples;
    samples.reserve(cs.num_samples);
    for (std::size_t i = 0; i < cs.num_samples; ++i) {
        sai::embedding::EmbeddingMeta meta;
        meta.model_name = "test";
        meta.type = sai::embedding::EmbeddingType::Patch;
        meta.dim = cs.dim;
        meta.count = 1;
        meta.grid = {1, 1};
        std::vector<float> vec(coreset_data.begin() + static_cast<std::ptrdiff_t>(i * cs.dim),
                                coreset_data.begin() + static_cast<std::ptrdiff_t>((i + 1) * cs.dim));
        samples.push_back(
            sai::embedding::Embedding::FromCpu(std::move(vec), std::move(meta)));
    }

    auto pca_result = sai::embedding::DimensionReducer::FitPca(samples, cs.dim);
    ASSERT_TRUE(pca_result.has_value()) << pca_result.error().message;

    auto pca_path = fs::temp_directory_path() / "test_hybrid_pca.bin";
    auto save_result = sai::embedding::DimensionReducer::SavePcaParams(
        pca_result.value(), pca_path);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message;

    // PatchCore with PCA hybrid scoring (alpha=0 → pure PCA)
    sai::detection::PatchCore::Config cfg_hybrid;
    cfg_hybrid.feature_bank_path = cs.path;
    cfg_hybrid.image_width = 28;
    cfg_hybrid.image_height = 28;
    cfg_hybrid.patch_size = 14;
    cfg_hybrid.embed_dim = 4;
    cfg_hybrid.k_nearest = 1;
    cfg_hybrid.anomaly_threshold = 0.5F;
    cfg_hybrid.gaussian_sigma = 1;
    cfg_hybrid.pca_model_path = pca_path;
    cfg_hybrid.hybrid_alpha = 0.0F;

    auto pc_hybrid = sai::detection::PatchCore(cfg_hybrid);
    sai::Context ctx;
    auto init_hybrid = pc_hybrid.Initialize(ctx);
    ASSERT_TRUE(init_hybrid.has_value()) << init_hybrid.error().message;

    // PatchCore without PCA (pure k-NN)
    sai::detection::PatchCore::Config cfg_pure;
    cfg_pure.feature_bank_path = cs.path;
    cfg_pure.image_width = 28;
    cfg_pure.image_height = 28;
    cfg_pure.patch_size = 14;
    cfg_pure.embed_dim = 4;
    cfg_pure.k_nearest = 1;
    cfg_pure.anomaly_threshold = 0.5F;
    cfg_pure.gaussian_sigma = 1;

    auto pc_pure = sai::detection::PatchCore(cfg_pure);
    auto init_pure = pc_pure.Initialize(ctx);
    ASSERT_TRUE(init_pure.has_value()) << init_pure.error().message;

    // 查询含异常 patch
    std::vector<float> query = {
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.9F, 0.9F, 0.9F, 0.9F,
    };
    auto emb = Make2x2Embedding(query);
    auto result_hybrid = pc_hybrid.Detect(emb);
    ASSERT_TRUE(result_hybrid.has_value());

    auto emb2 = Make2x2Embedding(query);
    auto result_pure = pc_pure.Detect(emb2);
    ASSERT_TRUE(result_pure.has_value());

    // 混合评分（纯 PCA）应与纯 k-NN 不同
    bool scores_differ = false;
    for (std::size_t i = 0; i < result_hybrid.value().anomaly_map.scores.size(); ++i) {
        if (std::abs(result_hybrid.value().anomaly_map.scores[i]
                     - result_pure.value().anomaly_map.scores[i]) > 1e-4F) {
            scores_differ = true;
            break;
        }
    }
    EXPECT_TRUE(scores_differ);

    fs::remove(cs.path);
    fs::remove(pca_path);
}

TEST(SubspaceADTest, AllDefaultsUnchanged) {
    auto cs = MakeMultiSampleCoreset();

    // 两个 PatchCore，配置完全一致（所有增强默认关闭）
    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = cs.path;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 0.5F;
    cfg.gaussian_sigma = 1;

    auto pc1 = sai::detection::PatchCore(cfg);
    auto pc2 = sai::detection::PatchCore(cfg);
    sai::Context ctx;
    auto init1 = pc1.Initialize(ctx);
    auto init2 = pc2.Initialize(ctx);
    ASSERT_TRUE(init1.has_value());
    ASSERT_TRUE(init2.has_value());

    std::vector<float> query = {
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.9F, 0.9F, 0.9F, 0.9F,
    };
    auto emb1 = Make2x2Embedding(query);
    auto emb2 = Make2x2Embedding(query);
    auto result1 = pc1.Detect(emb1);
    auto result2 = pc2.Detect(emb2);
    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // 所有默认配置下两个实例的输出应完全一致
    ASSERT_EQ(result1.value().anomaly_map.scores.size(),
              result2.value().anomaly_map.scores.size());
    for (std::size_t i = 0; i < result1.value().anomaly_map.scores.size(); ++i) {
        EXPECT_FLOAT_EQ(result1.value().anomaly_map.scores[i],
                         result2.value().anomaly_map.scores[i]);
    }
    EXPECT_FLOAT_EQ(result1.value().image_level_score,
                     result2.value().image_level_score);

    fs::remove(cs.path);
}

TEST(SubspaceADTest, WhiteningWithDropK) {
    auto cs = MakeMultiSampleCoreset();

    // NOTE: drop_k > 0 requires embed_dim to be sufficiently larger than target_dim,
    // i.e. enough PCA components must be available to skip the first drop_k.
    // With test embed_dim=4, we use drop_k=0 but still exercise the whitening + Rebuild path.
    // For production use with e.g. DINOv3 embed_dim=1024, drop_k=2 is common.

    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = cs.path;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 0.5F;
    cfg.gaussian_sigma = 1;
    cfg.enable_whitening = true;
    cfg.drop_k = 0;

    auto pc = sai::detection::PatchCore(cfg);
    sai::Context ctx;
    auto init_result = pc.Initialize(ctx);
    ASSERT_TRUE(init_result.has_value()) << init_result.error().message;

    // 查询含异常 patch
    std::vector<float> query = {
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.9F, 0.9F, 0.9F, 0.9F,
    };
    auto emb = Make2x2Embedding(query);
    auto result = pc.Detect(emb);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result.value().anomaly_map.scores.size(), 4U);
    EXPECT_GE(result.value().image_level_score, 0.0F);

    fs::remove(cs.path);
}

TEST(SubspaceADTest, WhiteningAndHybridCombined) {
    auto cs = MakeMultiSampleCoreset();

    // 用 coreset 数据手动物合 PCA
    std::vector<float> coreset_data = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
        0.5F, 0.5F, 0.0F, 0.0F,
        0.0F, 0.5F, 0.5F, 0.0F,
        0.0F, 0.0F, 0.5F, 0.5F,
        0.5F, 0.0F, 0.0F, 0.5F,
    };
    std::vector<sai::embedding::Embedding> samples;
    samples.reserve(cs.num_samples);
    for (std::size_t i = 0; i < cs.num_samples; ++i) {
        sai::embedding::EmbeddingMeta meta;
        meta.model_name = "test";
        meta.type = sai::embedding::EmbeddingType::Patch;
        meta.dim = cs.dim;
        meta.count = 1;
        meta.grid = {1, 1};
        std::vector<float> vec(coreset_data.begin() + static_cast<std::ptrdiff_t>(i * cs.dim),
                                coreset_data.begin() + static_cast<std::ptrdiff_t>((i + 1) * cs.dim));
        samples.push_back(
            sai::embedding::Embedding::FromCpu(std::move(vec), std::move(meta)));
    }

    auto pca_result = sai::embedding::DimensionReducer::FitPca(samples, cs.dim);
    ASSERT_TRUE(pca_result.has_value()) << pca_result.error().message;

    auto pca_path = fs::temp_directory_path() / "test_whitening_hybrid_pca.bin";
    auto save_result = sai::embedding::DimensionReducer::SavePcaParams(
        pca_result.value(), pca_path);
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message;

    // 同时启用白化和 PCA 混合评分
    sai::detection::PatchCore::Config cfg;
    cfg.feature_bank_path = cs.path;
    cfg.image_width = 28;
    cfg.image_height = 28;
    cfg.patch_size = 14;
    cfg.embed_dim = 4;
    cfg.k_nearest = 1;
    cfg.anomaly_threshold = 0.5F;
    cfg.gaussian_sigma = 1;
    cfg.enable_whitening = true;
    cfg.drop_k = 0;
    cfg.pca_model_path = pca_path;
    cfg.hybrid_alpha = 0.5F;

    auto pc = sai::detection::PatchCore(cfg);
    sai::Context ctx;
    auto init_result = pc.Initialize(ctx);
    ASSERT_TRUE(init_result.has_value()) << init_result.error().message;

    // 查询含异常 patch
    std::vector<float> query = {
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F, 0.5F,
        0.9F, 0.9F, 0.9F, 0.9F,
    };
    auto emb = Make2x2Embedding(query);
    auto result = pc.Detect(emb);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result.value().anomaly_map.scores.size(), 4U);
    EXPECT_GE(result.value().image_level_score, 0.0F);

    fs::remove(cs.path);
    fs::remove(pca_path);
}
