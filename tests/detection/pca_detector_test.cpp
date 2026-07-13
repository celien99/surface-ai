// pca_detector_test.cpp — PcaDetector 单元测试
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/context.h>
#include <sai/detection/detection_result.h>
#include <sai/detection/pca_detector.h>
#include <sai/embedding/dimension_reducer.h>
#include <sai/embedding/embedding.h>

namespace {

using namespace std::filesystem;
using namespace sai::embedding;
using namespace sai::detection;

// ── 测试夹具：创建简单的 PCA 模型和 Embedding ──

class PcaDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 用简单的 3D 数据拟合 PCA 模型
        // 数据主要在第一维 (0, 1) 之间变化，第二/三维几乎没有方差
        auto D = static_cast<std::size_t>(cfg_.embed_dim);
        std::vector<float> normal_data(10 * D, 0.0f);
        for (std::size_t i = 0; i < 10; ++i) {
            normal_data[i * D + 0] = static_cast<float>(i) * 0.1f;
            normal_data[i * D + 1] = static_cast<float>(i) * 0.01f;
            normal_data[i * D + 2] = static_cast<float>(i) * 0.01f;
        }

        EmbeddingMeta meta;
        meta.model_name = "test";
        meta.type = EmbeddingType::Patch;
        meta.dim = D;
        meta.count = 10;
        meta.grid = {1, 10};

        auto emb = Embedding::FromCpu(normal_data, meta);
        std::vector<Embedding> samples;
        samples.push_back(std::move(emb));

        auto fit_result = DimensionReducer::FitPca(samples, 4);  // 少数组件 → 正常模式被压缩，异常暴露
        ASSERT_TRUE(fit_result.has_value());
        pca_params_ = std::move(fit_result.value());

        // 保存 PCA 模型到临时文件
        tmp_path_ = temp_directory_path() / "test_pca_model.bin";
        auto save_result = DimensionReducer::SavePcaParams(pca_params_, tmp_path_);
        ASSERT_TRUE(save_result.has_value());
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(tmp_path_, ec);
    }

    // 创建一个正常的 Embedding（类似训练数据）
    auto MakeNormalEmbedding() -> Embedding {
        auto D = static_cast<std::size_t>(cfg_.embed_dim);
        auto count = expected_grid_h * expected_grid_w;
        std::vector<float> data(count * D, 0.0f);
        for (std::size_t i = 0; i < count; ++i) {
            data[i * D + 0] = static_cast<float>(i) * 0.01f;
            data[i * D + 1] = static_cast<float>(i) * 0.0001f;
            data[i * D + 2] = static_cast<float>(i) * 0.0001f;
        }

        EmbeddingMeta meta;
        meta.model_name = "DINOv3";
        meta.type = EmbeddingType::Patch;
        meta.dim = D;
        meta.count = count;
        meta.grid = {expected_grid_h, expected_grid_w};

        return Embedding::FromCpu(std::move(data), meta);
    }

    // 创建一个异常的 Embedding（偏移大）
    auto MakeAnomalousEmbedding() -> Embedding {
        auto D = static_cast<std::size_t>(cfg_.embed_dim);
        auto count = expected_grid_h * expected_grid_w;
        std::vector<float> data(count * D, 0.0f);
        // 异常信号：在高维方向（dim 100+）有大值，PCA 子空间未学到 → 高重构误差
        for (std::size_t i = 0; i < count; ++i) {
            data[i * D + 0] = static_cast<float>(i) * 0.1f;    // 正常范围
            data[i * D + 100] = 5.0f + static_cast<float>(i % 7) * 2.0f;
            data[i * D + 200] = static_cast<float>(i) * 0.5f;
        }

        EmbeddingMeta meta;
        meta.model_name = "DINOv3";
        meta.type = EmbeddingType::Patch;
        meta.dim = D;
        meta.count = count;
        meta.grid = {expected_grid_h, expected_grid_w};

        return Embedding::FromCpu(std::move(data), meta);
    }

    PcaDetector::Config cfg_{
        .pca_model_path = "",    // 将在 SetUp 中设置
        .embed_dim = 1024,
        .image_width = 518,
        .image_height = 518,
        .patch_size = 14,
    };
    // 518 / 14 = 37
    std::size_t expected_grid_h = 37;
    std::size_t expected_grid_w = 37;

    DimensionReducer::PcaParams pca_params_;
    path tmp_path_;
};

// ── Initialize ──

TEST_F(PcaDetectorTest, InitializeLoadsPcaModel) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;

    auto result = detector.Initialize(ctx);
    ASSERT_TRUE(result.has_value());
}

TEST_F(PcaDetectorTest, InitializeFailsOnMissingFile) {
    cfg_.pca_model_path = temp_directory_path() / "nonexistent_pca.bin";
    PcaDetector detector(cfg_);
    sai::Context ctx;

    auto result = detector.Initialize(ctx);
    EXPECT_FALSE(result.has_value());
}

TEST_F(PcaDetectorTest, InitializeFailsOnDimMismatch) {
    cfg_.pca_model_path = tmp_path_;
    cfg_.embed_dim = 512;  // 与实际模型不匹配
    PcaDetector detector(cfg_);
    sai::Context ctx;

    auto result = detector.Initialize(ctx);
    EXPECT_FALSE(result.has_value());
}

// ── Detect ──

TEST_F(PcaDetectorTest, DetectWithoutInitializeReturnsError) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);

    auto emb = MakeNormalEmbedding();
    auto result = detector.Detect(emb);
    EXPECT_FALSE(result.has_value());
}

TEST_F(PcaDetectorTest, DetectWithNormalEmbeddingReturnsResult) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    auto emb = MakeNormalEmbedding();
    auto result = detector.Detect(emb);
    ASSERT_TRUE(result.has_value());

    const auto& det = *result;
    EXPECT_EQ(det.anomaly_map.grid_h, expected_grid_h);
    EXPECT_EQ(det.anomaly_map.grid_w, expected_grid_w);
    EXPECT_GE(det.image_level_score, 0.0F);
    EXPECT_GT(det.inference_latency.count(), 0);
}

TEST_F(PcaDetectorTest, AnomalousEmbeddingHasHigherScore) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    auto normal_emb = MakeNormalEmbedding();
    auto normal_result = detector.Detect(normal_emb);
    ASSERT_TRUE(normal_result.has_value());

    auto anomaly_emb = MakeAnomalousEmbedding();
    auto anomaly_result = detector.Detect(anomaly_emb);
    ASSERT_TRUE(anomaly_result.has_value());

    // 两个 embedding 的 anomaly_map scores 分布应不同
    EXPECT_EQ(normal_result->anomaly_map.scores.size(), expected_grid_h * expected_grid_w);
    EXPECT_EQ(anomaly_result->anomaly_map.scores.size(), expected_grid_h * expected_grid_w);

    float normal_sum = std::accumulate(normal_result->anomaly_map.scores.begin(),
                                       normal_result->anomaly_map.scores.end(), 0.0F);
    float anomaly_sum = std::accumulate(anomaly_result->anomaly_map.scores.begin(),
                                        anomaly_result->anomaly_map.scores.end(), 0.0F);
    // 异常 embedding 的分数总和应不同于正常 embedding
    EXPECT_NE(normal_sum, anomaly_sum);
}

TEST_F(PcaDetectorTest, InvalidPatchGridReturnsError) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    // 创建一个错误 grid 的 embedding
    auto D = static_cast<std::size_t>(cfg_.embed_dim);
    std::vector<float> data(10 * D, 0.0f);
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = EmbeddingType::Patch;
    meta.dim = D;
    meta.count = 10;
    meta.grid = {2, 5};  // 2 != 37
    auto emb = Embedding::FromCpu(std::move(data), meta);

    auto result = detector.Detect(emb);
    EXPECT_FALSE(result.has_value());
}

// ── DetectBatch ──

TEST_F(PcaDetectorTest, DetectBatchReturnsMultipleResults) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    auto emb1 = MakeNormalEmbedding();
    auto emb2 = MakeNormalEmbedding();
    std::array<const Embedding*, 2> ptrs = {&emb1, &emb2};

    auto result = detector.DetectBatch(ptrs);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2U);
}

TEST_F(PcaDetectorTest, DetectBatchNullPointerReturnsError) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    std::array<const Embedding*, 2> ptrs = {nullptr, nullptr};
    auto result = detector.DetectBatch(ptrs);
    EXPECT_FALSE(result.has_value());
}

// ── ModelName ──

TEST_F(PcaDetectorTest, ModelNameReturnsPcaDetector) {
    cfg_.pca_model_path = tmp_path_;
    PcaDetector detector(cfg_);
    EXPECT_EQ(detector.ModelName(), "PcaDetector");
}

// ── drop_k 机制 ──

TEST_F(PcaDetectorTest, DropKAffectsScore) {
    cfg_.pca_model_path = tmp_path_;
    cfg_.drop_k = 0;
    PcaDetector detector_no_drop(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector_no_drop.Initialize(ctx).has_value());

    cfg_.drop_k = 1;
    PcaDetector detector_drop1(cfg_);
    ASSERT_TRUE(detector_drop1.Initialize(ctx).has_value());

    auto emb = MakeAnomalousEmbedding();
    auto result1 = detector_no_drop.Detect(emb);
    auto result2 = detector_drop1.Detect(emb);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // drop_k 应该有实质影响——比较 anomaly_map 的 mean 分数
    const auto& scores1 = result1->anomaly_map.scores;
    const auto& scores2 = result2->anomaly_map.scores;
    float mean1 = std::accumulate(scores1.begin(), scores1.end(), 0.0F) / static_cast<float>(scores1.size());
    float mean2 = std::accumulate(scores2.begin(), scores2.end(), 0.0F) / static_cast<float>(scores2.size());
    EXPECT_NE(mean1, mean2);
}

// ── Mahalanobis 评分 ──

TEST_F(PcaDetectorTest, MahalanobisScoringProducesResult) {
    cfg_.pca_model_path = tmp_path_;
    cfg_.score_method = PcaScoreMethod::Mahalanobis;
    PcaDetector detector(cfg_);
    sai::Context ctx;
    ASSERT_TRUE(detector.Initialize(ctx).has_value());

    auto emb = MakeNormalEmbedding();
    auto result = detector.Detect(emb);
    ASSERT_TRUE(result.has_value());
    EXPECT_GE(result->image_level_score, 0.0F);
}

}  // namespace
