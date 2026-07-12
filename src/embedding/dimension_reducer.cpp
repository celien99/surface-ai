// dimension_reducer.cpp — 批次 3.2 DimensionReducer PCA/Whitening 降维与 Pooling 实现
#include <algorithm>
#include <cmath>
#include <cstddef>
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

    // 处理数值退化情况：如果某特征值为零或噪声级，用该行的单位方向作为 fallback
    for (std::size_t t = 0; t < target_dim; ++t) {
        if (eigenvalues[t] < 1e-15f) {
            // 检查该行是否接近零
            float row_norm = 0.0f;
            for (std::size_t j = 0; j < input_dim; ++j) {
                row_norm += eigenvectors[t * input_dim + j] * eigenvectors[t * input_dim + j];
            }
            if (row_norm < 0.5f) {
                // 用单位向量填充
                std::fill(&eigenvectors[t * input_dim],
                          &eigenvectors[(t + 1) * input_dim], 0.0f);
                if (t < input_dim) {
                    eigenvectors[t * input_dim + t] = 1.0f;
                }
            }
        }
    }

    return PcaParams{
        .components = std::move(eigenvectors),
        .target_dim = target_dim,
        .mean = std::move(mean)};
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
// Constructors
// ============================================================================

DimensionReducer::DimensionReducer(PcaParams params) noexcept
    : is_pca_(true)
    , components_(std::move(params.components))
    , target_dim_(params.target_dim)
    , mean_(std::move(params.mean)) {
    if (target_dim_ > 0) {
        input_dim_ = components_.size() / target_dim_;
    } else {
        input_dim_ = 0;
    }
}

DimensionReducer::DimensionReducer(WhiteningParams params) noexcept
    : is_pca_(false)
    , components_(std::move(params.transform))
    , target_dim_(params.target_dim)
    , mean_() {
    if (target_dim_ > 0) {
        input_dim_ = components_.size() / target_dim_;
    } else {
        input_dim_ = 0;
    }
}

// ============================================================================
// ApplyReduce
// ============================================================================

auto DimensionReducer::ApplyReduce(const float* data, std::size_t count) const noexcept
    -> Result<std::vector<float>> {
    std::vector<float> result(count * target_dim_, 0.0f);

    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t t = 0; t < target_dim_; ++t) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < input_dim_; ++j) {
                float x = data[i * input_dim_ + j];
                if (is_pca_) {
                    x -= mean_[j];
                }
                acc += components_[t * input_dim_ + j] * x;
            }
            result[i * target_dim_ + t] = acc;
        }
    }

    return result;
}

// ============================================================================
// Reduce
// ============================================================================

auto DimensionReducer::Reduce(const Embedding& input) noexcept -> Result<Embedding> {
    if (input.IsOnGpu()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::Reduce: input is on GPU",
            std::source_location::current()});
    }

    const auto& meta = input.Meta();
    if (meta.dim != input_dim_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::Reduce: input dim (" + std::to_string(meta.dim)
                + ") != expected dim (" + std::to_string(input_dim_) + ")",
            std::source_location::current()});
    }

    if (meta.count == 0 || target_dim_ == 0) {
        // 空或零降维——返回空的缩减 Embedding
        EmbeddingMeta out_meta;
        out_meta.model_name = meta.model_name + "_reduced";
        out_meta.type = (meta.count <= 1) ? EmbeddingType::Global : EmbeddingType::Patch;
        out_meta.dim = target_dim_;
        out_meta.count = meta.count;
        if (meta.count > 1) {
            out_meta.grid = meta.grid;
        }
        return Embedding::FromCpu({}, out_meta);
    }

    auto result_data = ApplyReduce(input.Data(), meta.count);
    if (!result_data.has_value()) {
        return tl::make_unexpected(std::move(result_data.error()));
    }

    EmbeddingMeta out_meta;
    out_meta.model_name = meta.model_name + "_reduced";
    out_meta.type = (meta.count <= 1) ? EmbeddingType::Global : EmbeddingType::Patch;
    out_meta.dim = target_dim_;
    out_meta.count = meta.count;
    if (meta.count > 1) {
        out_meta.grid = meta.grid;
    }
    out_meta.inference_latency = meta.inference_latency;

    return Embedding::FromCpu(std::move(*result_data), out_meta);
}

// ============================================================================
// ReduceBatch
// ============================================================================

auto DimensionReducer::ReduceBatch(const std::vector<Embedding>& inputs) noexcept
    -> Result<std::vector<Embedding>> {
    std::vector<Embedding> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        auto reduced = Reduce(input);
        if (!reduced.has_value()) {
            return tl::make_unexpected(std::move(reduced.error()));
        }
        results.push_back(std::move(*reduced));
    }

    return results;
}

// ============================================================================
// Static Pool
// ============================================================================

auto DimensionReducer::Pool(const Embedding& input,
                              PoolingStrategy strategy) noexcept -> Result<Embedding> {
    if (input.IsOnGpu()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "DimensionReducer::Pool: input is on GPU",
            std::source_location::current()});
    }

    const auto& meta = input.Meta();
    if (meta.count == 0 || meta.dim == 0) {
        EmbeddingMeta out_meta;
        out_meta.model_name = meta.model_name + "_pooled";
        out_meta.type = EmbeddingType::Global;
        out_meta.dim = meta.dim;
        out_meta.count = 1;
        return Embedding::FromCpu({}, out_meta);
    }

    // 单向量无需池化
    if (meta.count == 1) {
        EmbeddingMeta out_meta = meta;
        out_meta.type = EmbeddingType::Global;
        out_meta.grid = {0, 0};
        out_meta.model_name = meta.model_name + "_pooled";
        return Embedding::FromCpu(
            std::vector<float>(input.Data(), input.Data() + meta.dim), out_meta);
    }

    const float* data = input.Data();
    std::vector<float> pooled(meta.dim);

    if (strategy == PoolingStrategy::Average) {
        // 逐元素均值
        std::copy(data, data + meta.dim, pooled.begin());
        for (std::size_t i = 1; i < meta.count; ++i) {
            for (std::size_t j = 0; j < meta.dim; ++j) {
                pooled[j] += data[i * meta.dim + j];
            }
        }
        float inv = 1.0f / static_cast<float>(meta.count);
        for (auto& v : pooled) {
            v *= inv;
        }
    } else {
        // Max pooling
        std::copy(data, data + meta.dim, pooled.begin());
        for (std::size_t i = 1; i < meta.count; ++i) {
            for (std::size_t j = 0; j < meta.dim; ++j) {
                float val = data[i * meta.dim + j];
                if (val > pooled[j]) {
                    pooled[j] = val;
                }
            }
        }
    }

    EmbeddingMeta out_meta;
    out_meta.model_name = meta.model_name + "_pooled";
    out_meta.type = EmbeddingType::Global;
    out_meta.dim = meta.dim;
    out_meta.count = 1;
    out_meta.grid = {0, 0};
    out_meta.inference_latency = meta.inference_latency;

    return Embedding::FromCpu(std::move(pooled), out_meta);
}

}  // namespace sai::embedding
