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

namespace {

// ============================================================================
// 内部辅助函数
// ============================================================================

// 收集样本中所有 Embedding 的浮点数据到扁平数组。
// 验证所有 Embedding 都在 CPU 端（IsOnGpu() == false）且 dim 一致。
// 返回 total_count × input_dim 的扁平数组，通过 out_input_dim 返回确认的维度。
[[nodiscard]] auto CollectSamples(const std::vector<Embedding>& samples,
                                   std::size_t& out_input_dim,
                                   std::size_t& out_total_count) -> Result<std::vector<float>> {
    if (samples.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer: no samples provided for fitting",
            std::source_location::current()});
    }

    // 检查并确定输入维度
    out_input_dim = 0;
    out_total_count = 0;
    bool dim_set = false;

    for (const auto& sample : samples) {
        if (sample.IsOnGpu()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Embedding_DimensionMismatch,
                "DimensionReducer: sample is on GPU, cannot fit on CPU",
                std::source_location::current()});
        }
        const auto& meta = sample.Meta();
        if (!dim_set) {
            out_input_dim = meta.dim;
            dim_set = true;
        } else if (meta.dim != out_input_dim) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Embedding_DimensionMismatch,
                "DimensionReducer: sample dim mismatch (" + std::to_string(meta.dim)
                    + " != " + std::to_string(out_input_dim) + ")",
                std::source_location::current()});
        }
        if (meta.dim > 0 && meta.count > 0) {
            out_total_count += meta.count;
        }
    }

    if (out_total_count == 0 || out_input_dim == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer: all samples have zero count or zero dimension",
            std::source_location::current()});
    }

    // 收集数据
    std::vector<float> data;
    data.reserve(out_total_count * out_input_dim);
    for (const auto& sample : samples) {
        const float* src = sample.Data();
        std::size_t n = sample.Meta().count;
        std::copy(src, src + n * out_input_dim, std::back_inserter(data));
    }
    return data;
}

// 计算均值向量：data 为 count × dim 的扁平数组，输出 dim 维均值。
[[nodiscard]] auto ComputeMean(const float* data, std::size_t count,
                                std::size_t dim) -> std::vector<float> {
    std::vector<float> mean(dim, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            mean[j] += data[i * dim + j];
        }
    }
    float inv = 1.0f / static_cast<float>(count);
    for (auto& m : mean) {
        m *= inv;
    }
    return mean;
}

// 中心化数据：从 data（count × dim）逐行减去 mean，写入 centered。
void CenterData(const float* data, std::size_t count, std::size_t dim,
                const float* mean, float* centered) {
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            centered[i * dim + j] = data[i * dim + j] - mean[j];
        }
    }
}

// 计算协方差矩阵（count × dim 的中心化数据 → dim × dim 协方差矩阵，行优先）。
void ComputeCovariance(const float* centered, std::size_t count,
                        std::size_t dim, float* cov) {
    std::fill(cov, cov + dim * dim, 0.0f);
    for (std::size_t k = 0; k < count; ++k) {
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                cov[i * dim + j] += centered[k * dim + i] * centered[k * dim + j];
            }
        }
    }
    float inv = 1.0f / static_cast<float>(count);
    for (std::size_t i = 0; i < dim * dim; ++i) {
        cov[i] *= inv;
    }
}

// 幂迭代法求最大特征对：(v, λ) = argmax v^T * C * v / v^T * v。
// mat 为 dim × dim 对称矩阵，行优先。
// 返回 {eigenvector (dim), eigenvalue}。
[[nodiscard]] auto PowerIteration(const float* mat, std::size_t dim,
                                   std::size_t max_iters = 200,
                                   float tol = 1e-8f)
    -> std::pair<std::vector<float>, float> {
    // 初始向量：均匀方向 + 小随机扰动的确定性替代
    std::vector<float> v(dim, 0.0f);
    v[0] = 1.0f;
    float v_norm = 1.0f;

    float eigenvalue = 0.0f;
    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        // w = mat * v
        std::vector<float> w(dim, 0.0f);
        for (std::size_t i = 0; i < dim; ++i) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < dim; ++j) {
                acc += mat[i * dim + j] * v[j];
            }
            w[i] = acc;
        }

        // Rayleigh 商：λ = v^T * w
        eigenvalue = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            eigenvalue += v[i] * w[i];
        }

        if (eigenvalue < 0.0f) {
            eigenvalue = 0.0f;
        }

        // 归一化
        float norm = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            norm += w[i] * w[i];
        }
        norm = std::sqrt(norm);
        if (norm < tol) {
            // 零向量：剩余方差为零
            std::fill(v.begin(), v.end(), 0.0f);
            if (dim > 0) v[0] = 1.0f;
            eigenvalue = 0.0f;
            break;
        }
        for (std::size_t i = 0; i < dim; ++i) {
            w[i] /= norm;
        }

        // 检查收敛：||w - v|| 或 ||w + v||
        float diff = 0.0f;
        float diff_neg = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            float d = w[i] - v[i];
            float dn = w[i] + v[i];
            diff += d * d;
            diff_neg += dn * dn;
        }
        v.swap(w);
        if (diff < tol || diff_neg < tol) break;
    }

    return {v, eigenvalue};
}

// 计算特征分解：通过幂迭代 + 缩减去顶法获取 top_k 个特征向量/值。
// cov 为 dim × dim 对称矩阵（被修改——缩减去顶法）。
// 返回 target_dim 个特征向量（target_dim × dim），和特征值数组（target_dim）。
[[nodiscard]] auto ComputeTopEigen(float* cov, std::size_t dim,
                                    std::size_t target_dim)
    -> std::pair<std::vector<float>, std::vector<float>> {
    std::vector<float> eigenvectors(target_dim * dim, 0.0f);
    std::vector<float> eigenvalues(target_dim, 0.0f);

    for (std::size_t t = 0; t < target_dim; ++t) {
        auto [ev, lambda] = PowerIteration(cov, dim);
        eigenvalues[t] = lambda;

        // 复制特征向量到输出
        std::copy(ev.begin(), ev.end(), &eigenvectors[t * dim]);

        // 缩减去顶：cov -= λ * v * v^T
        if (lambda > 1e-12f) {
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    cov[i * dim + j] -= lambda * ev[i] * ev[j];
                }
            }
        }
    }

    return {eigenvectors, eigenvalues};
}

// 退化处理：如果某主成分为零或噪声级，用单位向量回退
void FixDegenerateComponents(std::vector<float>& eigenvectors, std::vector<float>& eigenvalues,
                              std::size_t target_dim, std::size_t input_dim) {
    for (std::size_t t = 0; t < target_dim; ++t) {
        if (eigenvalues[t] < 1e-15f) {
            float row_norm = 0.0f;
            for (std::size_t j = 0; j < input_dim; ++j) {
                row_norm += eigenvectors[t * input_dim + j] * eigenvectors[t * input_dim + j];
            }
            if (row_norm < 0.5f) {
                std::fill(&eigenvectors[t * input_dim], &eigenvectors[(t + 1) * input_dim], 0.0f);
                if (t < input_dim) {
                    eigenvectors[t * input_dim + t] = 1.0f;
                }
            }
        }
    }
}

}  // namespace

// ============================================================================
// Static factory: FitPca
// ============================================================================

auto DimensionReducer::FitPca(const std::vector<Embedding>& samples,
                                std::size_t target_dim) noexcept -> Result<PcaParams> {
    std::size_t input_dim = 0;
    std::size_t total_count = 0;

    auto data_result = CollectSamples(samples, input_dim, total_count);
    if (!data_result.has_value()) {
        return tl::make_unexpected(std::move(data_result.error()));
    }
    const auto& data = *data_result;

    if (target_dim > input_dim) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::FitPca: target_dim (" + std::to_string(target_dim)
                + ") > input_dim (" + std::to_string(input_dim) + ")",
            std::source_location::current()});
    }

    // 1. 计算均值
    auto mean = ComputeMean(data.data(), total_count, input_dim);

    // 2. 中心化
    std::vector<float> centered(total_count * input_dim);
    CenterData(data.data(), total_count, input_dim, mean.data(), centered.data());

    // 3. 计算协方差矩阵
    std::vector<float> cov(input_dim * input_dim);
    ComputeCovariance(centered.data(), total_count, input_dim, cov.data());

    // 4. 特征分解——获取 top target_dim 个特征对
    // cov 在 ComputeTopEigen 中被修改（deflation），但已不需要原值
    auto [eigenvectors, eigenvalues] = ComputeTopEigen(cov.data(), input_dim, target_dim);

    FixDegenerateComponents(eigenvectors, eigenvalues, target_dim, input_dim);

    return PcaParams{
        .components = std::move(eigenvectors),
        .target_dim = target_dim,
        .mean = std::move(mean),
        .eigvals = std::move(eigenvalues)};
}

// ============================================================================
// Static factory: FitWhitening
// ============================================================================

auto DimensionReducer::FitWhitening(const std::vector<Embedding>& samples,
                                      std::size_t target_dim) noexcept -> Result<WhiteningParams> {
    std::size_t input_dim = 0;
    std::size_t total_count = 0;

    auto data_result = CollectSamples(samples, input_dim, total_count);
    if (!data_result.has_value()) {
        return tl::make_unexpected(std::move(data_result.error()));
    }
    const auto& data = *data_result;

    if (target_dim > input_dim) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::FitWhitening: target_dim (" + std::to_string(target_dim)
                + ") > input_dim (" + std::to_string(input_dim) + ")",
            std::source_location::current()});
    }

    // 1. 计算均值（用于中心化后再计算协方差）
    auto mean = ComputeMean(data.data(), total_count, input_dim);

    // 2. 中心化
    std::vector<float> centered(total_count * input_dim);
    CenterData(data.data(), total_count, input_dim, mean.data(), centered.data());

    // 3. 计算协方差矩阵
    std::vector<float> cov(input_dim * input_dim);
    ComputeCovariance(centered.data(), total_count, input_dim, cov.data());

    // 4. 特征分解
    auto [eigenvectors, eigenvalues] = ComputeTopEigen(cov.data(), input_dim, target_dim);

    // 5. 构建白化变换矩阵
    std::vector<float> transform(target_dim * input_dim, 0.0f);
    for (std::size_t t = 0; t < target_dim; ++t) {
        float scale = (eigenvalues[t] > 1e-12f)
            ? 1.0f / std::sqrt(eigenvalues[t])
            : 0.0f;
        for (std::size_t j = 0; j < input_dim; ++j) {
            transform[t * input_dim + j] = eigenvectors[t * input_dim + j] * scale;
        }
    }

    return WhiteningParams{
        .transform = std::move(transform),
        .target_dim = target_dim};
}


// ============================================================================
// Static FitPcaStreaming — 流式两遍 PCA 拟合
// ============================================================================

auto DimensionReducer::FitPcaStreaming(
    std::function<std::function<std::vector<float>()>()> make_generator,
    std::size_t D, std::size_t total_N, std::size_t target_k) noexcept -> Result<PcaParams> {
    if (D == 0 || total_N == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::FitPcaStreaming: D and total_N must be > 0",
            std::source_location::current()});
    }

    if (target_k == 0) {
        target_k = D;
    }
    if (target_k > D) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::FitPcaStreaming: target_k (" + std::to_string(target_k)
                + ") > D (" + std::to_string(D) + ")",
            std::source_location::current()});
    }

    // ── Pass 1: 计算均值 ──
    std::vector<double> mean_accum(D, 0.0);
    std::size_t seen = 0;
    {
        auto gen = make_generator();
        while (true) {
            auto batch = gen();
            if (batch.empty()) break;
            auto batch_n = batch.size() / D;
            for (std::size_t i = 0; i < batch_n; ++i) {
                for (std::size_t j = 0; j < D; ++j) {
                    mean_accum[j] += static_cast<double>(batch[i * D + j]);
                }
            }
            seen += batch_n;
        }
    }
    if (seen != total_N) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::FitPcaStreaming: expected " + std::to_string(total_N)
                + " vectors but saw " + std::to_string(seen),
            std::source_location::current()});
    }

    std::vector<float> mean(D);
    double inv_N = 1.0 / static_cast<double>(total_N);
    for (std::size_t j = 0; j < D; ++j) {
        mean[j] = static_cast<float>(mean_accum[j] * inv_N);
    }

    // ── Pass 2: 计算协方差矩阵 ──
    std::vector<double> cov(D * D, 0.0);
    {
        auto gen = make_generator();  // 重新创建 generator，从头遍历
        while (true) {
            auto batch = gen();
            if (batch.empty()) break;
            auto batch_n = batch.size() / D;
            for (std::size_t i = 0; i < batch_n; ++i) {
                for (std::size_t p = 0; p < D; ++p) {
                    double xp = static_cast<double>(batch[i * D + p]) - static_cast<double>(mean[p]);
                    for (std::size_t q = p; q < D; ++q) {
                        double xq = static_cast<double>(batch[i * D + q]) - static_cast<double>(mean[q]);
                        cov[p * D + q] += xp * xq;
                    }
                }
            }
        }
    }
    // 对称填充 + 归一化
    double inv_Nm1 = 1.0 / static_cast<double>(total_N - 1);
    for (std::size_t p = 0; p < D; ++p) {
        for (std::size_t q = p; q < D; ++q) {
            double v = cov[p * D + q] * inv_Nm1;
            cov[p * D + q] = v;
            cov[q * D + p] = v;
        }
    }

    // ── 特征分解 ──
    // 将 double cov 转成 float 用于 ComputeTopEigen
    std::vector<float> cov_f(D * D);
    for (std::size_t i = 0; i < D * D; ++i) {
        cov_f[i] = static_cast<float>(cov[i]);
    }
    auto [eigenvectors, eigenvalues] = ComputeTopEigen(cov_f.data(), D, target_k);

    FixDegenerateComponents(eigenvectors, eigenvalues, target_k, D);

    return PcaParams{
        .components = std::move(eigenvectors),
        .target_dim = target_k,
        .mean = std::move(mean),
        .eigvals = std::move(eigenvalues)};
}


}  // namespace sai::embedding
