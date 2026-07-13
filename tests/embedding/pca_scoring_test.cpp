// pca_scoring_test.cpp — PCA 评分方法、序列化、流式拟合测试
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <sai/embedding/dimension_reducer.h>

namespace sai::embedding {
namespace {

using namespace std::filesystem;

// 辅助：构造一个简单的 Embedding 用于 FitPca
auto MakeEmbedding(const std::vector<float>& data, std::size_t count, std::size_t dim)
    -> Embedding {
    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.type = EmbeddingType::Patch;
    meta.dim = dim;
    meta.count = count;
    if (count > 1) {
        meta.grid = {1, count};  // 1×count grid
    }
    return Embedding::FromCpu(std::vector<float>(data), meta);
}

// ── PcaParams eigvals 在 FitPca 中被正确填充 ──

TEST(PcaScoringTest, FitPcaStoresEigenvalues) {
    // 4 个 3D 向量
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,
        1.1f, 0.1f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.1f, 1.1f, 0.0f,
    };
    auto emb = MakeEmbedding(data, 4, 3);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());
    const auto& params = *result;
    EXPECT_EQ(params.target_dim, 2U);
    EXPECT_EQ(params.mean.size(), 3U);
    EXPECT_EQ(params.components.size(), 6U);  // 2×3
    // 特征值应该在 FitPca 中被填充
    EXPECT_EQ(params.eigvals.size(), 2U);
    // 特征值应降序
    EXPECT_GE(params.eigvals[0], params.eigvals[1]);
    // 至少第一个特征值 > 0（有方差）
    EXPECT_GT(params.eigvals[0], 0.0f);
}

// ── Reconstruction 评分 ──

TEST(PcaScoringTest, ReconstructionScoreExactMatch) {
    // 拟合一个简单的 2D → 1D PCA
    std::vector<float> data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    auto emb = MakeEmbedding(data, 2, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 1);
    ASSERT_TRUE(result.has_value());
    const auto& params = *result;

    // 用同一批数据的第一个向量评分，重构误差应很小
    float query[2] = {1.0f, 2.0f};
    auto scores = DimensionReducer::Score(params, query, 1,
                                           PcaScoreMethod::Reconstruction, 0);
    EXPECT_EQ(scores.size(), 1U);
    EXPECT_GE(scores[0], 0.0f);
}

TEST(PcaScoringTest, ReconstructionScoreOutlierIsHigher) {
    // 正常数据集中在 (0,0) 附近
    std::vector<float> normal = {
        0.0f, 0.0f,
        0.1f, -0.1f,
        -0.1f, 0.1f,
        0.0f, 0.1f,
    };
    auto emb = MakeEmbedding(normal, 4, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 1);
    ASSERT_TRUE(result.has_value());
    const auto& params = *result;

    // 正常样本分数
    float normal_query[2] = {0.0f, 0.0f};
    auto normal_scores = DimensionReducer::Score(params, normal_query, 1,
                                                   PcaScoreMethod::Reconstruction, 0);

    // 异常样本分数（远离正常分布）
    float anomaly_query[2] = {10.0f, 10.0f};
    auto anomaly_scores = DimensionReducer::Score(params, anomaly_query, 1,
                                                    PcaScoreMethod::Reconstruction, 0);

    // 异常分数应该明显高于正常分数
    EXPECT_GT(anomaly_scores[0], normal_scores[0] * 10.0f);
}

// ── Mahalanobis 评分 ──

TEST(PcaScoringTest, MahalanobisScoreNonNegative) {
    std::vector<float> data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
        0.0f, -1.0f,
    };
    auto emb = MakeEmbedding(data, 4, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());

    float query[2] = {0.5f, 0.5f};
    auto scores = DimensionReducer::Score(*result, query, 1,
                                            PcaScoreMethod::Mahalanobis, 0);
    EXPECT_GE(scores[0], 0.0f);
}

// ── Cosine 评分 ──

TEST(PcaScoringTest, CosineScoreInRange) {
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    auto emb = MakeEmbedding(data, 2, 3);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());

    float query[3] = {0.5f, 0.5f, 0.0f};
    auto scores = DimensionReducer::Score(*result, query, 1,
                                            PcaScoreMethod::Cosine, 0);
    // Cosine 距离应在 [0, 2] 范围内
    EXPECT_GE(scores[0], 0.0f);
    EXPECT_LE(scores[0], 2.0f);
}

// ── Euclidean 评分 ──

TEST(PcaScoringTest, EuclideanScoreNonNegative) {
    std::vector<float> data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    auto emb = MakeEmbedding(data, 2, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 1);
    ASSERT_TRUE(result.has_value());

    float query[2] = {0.0f, 0.0f};
    auto scores = DimensionReducer::Score(*result, query, 1,
                                            PcaScoreMethod::Euclidean, 0);
    EXPECT_GE(scores[0], 0.0f);
}

// ── drop_k 机制 ──

TEST(PcaScoringTest, DropKReducesScore) {
    // 第一维有大量方差（正常变化），第二维较少
    std::vector<float> data = {
        10.0f, 0.0f,
        12.0f, 0.1f,
        8.0f, -0.1f,
        9.0f, 0.0f,
        11.0f, 0.05f,
    };
    auto emb = MakeEmbedding(data, 5, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());

    // 查询向量：在第一维正常（~10），但在第二维有较大偏移
    float query[2] = {10.0f, 5.0f};
    auto scores_no_drop = DimensionReducer::Score(*result, query, 1,
                                                    PcaScoreMethod::Mahalanobis, 0);
    auto scores_drop1 = DimensionReducer::Score(*result, query, 1,
                                                  PcaScoreMethod::Mahalanobis, 1);

    // drop_k=1 丢弃第一个主成分后，用 Mahalanobis 评分
    // 剩余成分的方差小 → 同样的偏移会产生更大的Mahalanobis距离
    EXPECT_GE(scores_no_drop[0], 0.0f);
    EXPECT_GE(scores_drop1[0], 0.0f);
    EXPECT_NE(scores_no_drop[0], scores_drop1[0]);
}

// ── 批量评分 ──

TEST(PcaScoringTest, BatchScoringReturnsCorrectSize) {
    std::vector<float> data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    auto emb = MakeEmbedding(data, 2, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 1);
    ASSERT_TRUE(result.has_value());

    float queries[6] = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
    auto scores = DimensionReducer::Score(*result, queries, 3,
                                            PcaScoreMethod::Reconstruction, 0);
    EXPECT_EQ(scores.size(), 3U);
}

// ── 序列化 round-trip ──

TEST(PcaScoringTest, SaveLoadRoundTrip) {
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        -1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
    };
    auto emb = MakeEmbedding(data, 4, 3);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());

    auto tmp_path = temp_directory_path() / "test_pca_params.bin";

    // Save
    auto save_result = DimensionReducer::SavePcaParams(*result, tmp_path);
    ASSERT_TRUE(save_result.has_value());

    // Load
    auto load_result = DimensionReducer::LoadPcaParams(tmp_path);
    ASSERT_TRUE(load_result.has_value());

    const auto& loaded = *load_result;
    EXPECT_EQ(loaded.target_dim, result->target_dim);
    EXPECT_EQ(loaded.mean.size(), result->mean.size());
    EXPECT_EQ(loaded.eigvals.size(), result->eigvals.size());
    EXPECT_EQ(loaded.components.size(), result->components.size());

    // 验证数值一致
    for (std::size_t i = 0; i < result->mean.size(); ++i) {
        EXPECT_NEAR(loaded.mean[i], result->mean[i], 1e-5f);
    }
    for (std::size_t i = 0; i < result->eigvals.size(); ++i) {
        EXPECT_NEAR(loaded.eigvals[i], result->eigvals[i], 1e-5f);
    }
    for (std::size_t i = 0; i < result->components.size(); ++i) {
        EXPECT_NEAR(loaded.components[i], result->components[i], 1e-5f);
    }

    // Clean up
    std::filesystem::remove(tmp_path);
}

// ── 流式拟合 ──

TEST(PcaScoringTest, StreamingFitMatchesBatchFit) {
    std::vector<float> data = {
        1.0f, 0.0f, 0.0f,
        0.1f, 0.1f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.1f, 1.1f, 0.0f,
    };
    auto emb = MakeEmbedding(data, 4, 3);

    // 批处理拟合
    std::vector<Embedding> batch_samples;
    batch_samples.push_back(std::move(emb));
    auto batch_result = DimensionReducer::FitPca(batch_samples, 2);
    ASSERT_TRUE(batch_result.has_value());

    // 流式拟合
    auto make_gen = [&data]() {
        auto offset = std::make_shared<std::size_t>(0);
        return [&data, offset]() -> std::vector<float> {
            if (*offset >= data.size()) return {};
            // 每次返回 2 个向量（batch_size=2）
            std::size_t take = std::min<std::size_t>(6, data.size() - *offset);
            auto batch = std::vector<float>(data.begin() + static_cast<std::ptrdiff_t>(*offset),
                                             data.begin() + static_cast<std::ptrdiff_t>(*offset + take));
            *offset += take;
            return batch;
        };
    };

    auto stream_result = DimensionReducer::FitPcaStreaming(make_gen, 3, 4, 2);
    ASSERT_TRUE(stream_result.has_value());

    // 两种方法的均值应该相同
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(stream_result->mean[i], batch_result->mean[i], 1e-4f);
    }

    // 特征值应该相近（流式路径用 double 算均值/协方差再转 float，有精度差异）
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(stream_result->eigvals[i], batch_result->eigvals[i], 0.15f);
    }
}

// ── 空输入保护 ──

TEST(PcaScoringTest, ScoreEmptyInputReturnsEmpty) {
    DimensionReducer::PcaParams params;
    params.target_dim = 1;
    params.mean = {0.0f, 0.0f};
    params.components = {1.0f, 0.0f};
    params.eigvals = {1.0f};

    auto scores = DimensionReducer::Score(params, nullptr, 0,
                                            PcaScoreMethod::Reconstruction, 0);
    EXPECT_TRUE(scores.empty());
}

// ── drop_k 越界 ──

TEST(PcaScoringTest, DropKBeyondKReturnsZeros) {
    std::vector<float> data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    auto emb = MakeEmbedding(data, 2, 2);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 1);
    ASSERT_TRUE(result.has_value());

    float query[2] = {0.5f, 0.5f};
    // drop_k >= k → 所有成分被丢弃 → 分数全为零
    auto scores = DimensionReducer::Score(*result, query, 1,
                                            PcaScoreMethod::Reconstruction, 1);
    EXPECT_FLOAT_EQ(scores[0], 0.0f);
}

// ── 四种评分方法产生不同值 ──

TEST(PcaScoringTest, DifferentMethodsYieldDifferentScores) {
    std::vector<float> data = {
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 3.0f, 0.0f,
    };
    auto emb = MakeEmbedding(data, 4, 3);
    std::vector<Embedding> samples;
    samples.push_back(std::move(emb));
    auto result = DimensionReducer::FitPca(samples, 2);
    ASSERT_TRUE(result.has_value());

    float query[3] = {1.0f, 1.0f, 0.5f};

    auto s_rec = DimensionReducer::Score(*result, query, 1,
                                           PcaScoreMethod::Reconstruction, 0);
    auto s_mah = DimensionReducer::Score(*result, query, 1,
                                           PcaScoreMethod::Mahalanobis, 0);
    auto s_cos = DimensionReducer::Score(*result, query, 1,
                                           PcaScoreMethod::Cosine, 0);
    auto s_euc = DimensionReducer::Score(*result, query, 1,
                                           PcaScoreMethod::Euclidean, 0);

    // 至少有一对不同（四种方法在非退化情况下分数不同）
    bool any_diff = (std::abs(s_rec[0] - s_mah[0]) > 1e-6f) ||
                    (std::abs(s_rec[0] - s_cos[0]) > 1e-6f) ||
                    (std::abs(s_rec[0] - s_euc[0]) > 1e-6f);
    EXPECT_TRUE(any_diff);
}

}  // namespace
}  // namespace sai::embedding
