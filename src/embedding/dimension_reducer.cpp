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
