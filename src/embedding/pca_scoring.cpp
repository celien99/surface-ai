// dimension_reducer.cpp — 批次 3.2 DimensionReducer PCA/Whitening 降维、评分、序列化与 Pooling 实现
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <sai/embedding/dimension_reducer.h>

namespace sai::embedding {

// ============================================================================
// Static Score — PCA 异常评分
// ============================================================================

namespace {

// 重构：X_recon = (X0 @ C) @ Cᵀ + mu（仅 Cosine 评分使用）
auto PcaReconstruct(const float* X, std::size_t N, const DimensionReducer::PcaParams& pca,
                    std::size_t drop_k) -> std::vector<float> {
    auto D = static_cast<std::size_t>(pca.mean.size());
    auto k = pca.target_dim;
    const float* mu = pca.mean.data();
    const float* C = pca.components.data();  // k × D row-major

    std::vector<float> X_recon(N * D);
    for (std::size_t i = 0; i < N; ++i) {
        // Z = (X_i - mu) @ Cᵀ  →  Z shape: k
        for (std::size_t t = 0; t < k; ++t) {
            float z = 0.0f;
            if (t >= drop_k) {
                for (std::size_t j = 0; j < D; ++j) {
                    z += (X[i * D + j] - mu[j]) * C[t * D + j];
                }
            }
            // X_recon = Z @ C + mu
            for (std::size_t j = 0; j < D; ++j) {
                X_recon[i * D + j] += z * C[t * D + j];
            }
        }
        for (std::size_t j = 0; j < D; ++j) {
            X_recon[i * D + j] += mu[j];
        }
    }
    return X_recon;
}

}  // namespace

auto DimensionReducer::Score(const PcaParams& params, const float* X,
                               std::size_t N, PcaScoreMethod method,
                               std::size_t drop_k) noexcept -> std::vector<float> {
    if (N == 0) return {};

    auto D = static_cast<std::size_t>(params.mean.size());
    auto k = params.target_dim;
    const float* mu = params.mean.data();
    const float* C = params.components.data();

    if (drop_k >= k) {
        // 所有成分被丢弃 → 分数全为零
        return std::vector<float>(N, 0.0f);
    }

    std::vector<float> scores(N, 0.0f);

    if (method == PcaScoreMethod::Reconstruction) {
        // ||X - X_recon||² = ||X0||² - ||Z||²（C 为正交基，该等式精确成立）
        // 无需构建完整 X_recon，避免 O(N×D) 额外内存分配
        for (std::size_t i = 0; i < N; ++i) {
            float x0_norm_sq = 0.0f;
            float z_norm_sq = 0.0f;
            for (std::size_t j = 0; j < D; ++j) {
                float x0 = X[i * D + j] - mu[j];
                x0_norm_sq += x0 * x0;
            }
            for (std::size_t t = drop_k; t < k; ++t) {
                float z = 0.0f;
                for (std::size_t j = 0; j < D; ++j) {
                    z += (X[i * D + j] - mu[j]) * C[t * D + j];
                }
                z_norm_sq += z * z;
            }
            scores[i] = std::max(0.0f, x0_norm_sq - z_norm_sq);
        }
    } else if (method == PcaScoreMethod::Mahalanobis) {
        const float* eig = params.eigvals.data();
        float eps = 1e-6f;
        for (std::size_t i = 0; i < N; ++i) {
            float s = 0.0f;
            for (std::size_t t = drop_k; t < k; ++t) {
                float z = 0.0f;
                for (std::size_t j = 0; j < D; ++j) {
                    z += (X[i * D + j] - mu[j]) * C[t * D + j];
                }
                float inv_lambda = 1.0f / (eig[t] + eps);
                s += z * z * inv_lambda;
            }
            scores[i] = s;
        }
    } else if (method == PcaScoreMethod::Euclidean) {
        for (std::size_t i = 0; i < N; ++i) {
            float s = 0.0f;
            for (std::size_t t = drop_k; t < k; ++t) {
                float z = 0.0f;
                for (std::size_t j = 0; j < D; ++j) {
                    z += (X[i * D + j] - mu[j]) * C[t * D + j];
                }
                s += z * z;
            }
            scores[i] = s;
        }
    } else if (method == PcaScoreMethod::Cosine) {
        float eps = 1e-8f;
        auto X_recon = PcaReconstruct(X, N, params, drop_k);
        for (std::size_t i = 0; i < N; ++i) {
            // 原地 L2 归一化 + 点积 = cosine similarity
            float x_norm = eps, xr_norm = eps, dot = 0.0f;
            for (std::size_t j = 0; j < D; ++j) {
                x_norm += X[i * D + j] * X[i * D + j];
                xr_norm += X_recon[i * D + j] * X_recon[i * D + j];
                dot += X[i * D + j] * X_recon[i * D + j];
            }
            float sim = dot / (std::sqrt(x_norm) * std::sqrt(xr_norm));
            sim = std::max(-1.0f, std::min(1.0f, sim));
            scores[i] = 1.0f - sim;
        }
    }

    return scores;
}


}  // namespace sai::embedding
