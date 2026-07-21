// patch_core.cpp — PatchCore::Detect 完整管线实现
#include <sai/detection/patch_core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <source_location>
#include <unordered_map>
#include <vector>

#include <sai/detection/feature_bank.h>
#include <sai/detection/post_process_utils.h>
#include <sai/core/rect.h>
#include <sai/embedding/dimension_reducer.h>

#if defined(SAI_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

namespace sai::detection {
namespace {

// 白化变换：out[N * wp.target_dim] = W @ in[N * input_dim]
// W = wp.transform (target_dim × input_dim, row-major)
auto ApplyWhitening(const float* vectors, std::size_t count,
                    std::size_t input_dim,
                    const sai::embedding::DimensionReducer::WhiteningParams& wp) noexcept
    -> std::vector<float> {
    auto target_dim = wp.target_dim;
    std::vector<float> result(count * target_dim, 0.0F);
    for (std::size_t n = 0; n < count; ++n) {
        const float* in = vectors + n * input_dim;
        float* out = result.data() + n * target_dim;
        for (std::size_t t = 0; t < target_dim; ++t) {
            float acc = 0.0F;
            const float* row = wp.transform.data() + t * input_dim;
            for (std::size_t j = 0; j < input_dim; ++j) {
                acc += row[j] * in[j];
            }
            out[t] = acc;
        }
    }
    return result;
}

// 自适应阈值：对 coreset 进行自检，计算 k-NN 距离的 (1-target_fpr) 分位数
// 同时返回所有自查询距离的最大值作为全局归一化参考值。
auto ComputeAdaptiveThreshold(const FeatureBank& bank,
                              float target_fpr,
                              std::size_t k,
                              float& out_ref_dist) noexcept -> float {
    auto num_samples = bank.NumSamples();
    if (num_samples == 0) return 0.0F;

    auto dim = bank.Dim();
    auto all_vecs = bank.ExtractAllVectors();
    std::vector<float> nn_dists(num_samples);

    auto search_k = std::min<std::size_t>(k + 1, num_samples);
    auto dists = bank.Search(all_vecs.data(), num_samples, search_k);
    float max_self_dist = 0.0F;
    for (std::size_t i = 0; i < num_samples; ++i) {
        nn_dists[i] = dists[i * search_k + search_k - 1];
        if (nn_dists[i] > max_self_dist) max_self_dist = nn_dists[i];
    }

    std::sort(nn_dists.begin(), nn_dists.end());

    auto idx = static_cast<std::size_t>((1.0F - target_fpr) * static_cast<float>(num_samples - 1));
    if (idx >= num_samples) idx = num_samples - 1;

    out_ref_dist = max_self_dist;
    return nn_dists[idx];
}

}  // namespace

// ── PatchCore::Initialize ───────────────────────────────────────────

auto PatchCore::Initialize(sai::Context& /*ctx*/) noexcept -> Result<void> {
    // If FeatureBank was injected via SetFeatureBank(), skip LoadFromFile.
    if (feature_bank_) {
        // Already loaded — proceed to optional whitening/threshold/PCA.
    } else if (!cfg_.feature_bank_path.empty()) {
        auto load_result = FeatureBank::LoadFromFile(cfg_.feature_bank_path, cfg_.embed_dim);
        if (!load_result) return tl::make_unexpected(load_result.error());
        feature_bank_ = std::make_unique<FeatureBank>(std::move(*load_result));
    }
    // If neither injected nor configured, feature_bank_ stays null →
    // Detect() will return Detection_FeatureBankLoadFailed.

    return Result<void>{}
        .and_then([this]() -> Result<void> {
            if (!feature_bank_) return Result<void>{};
            if (!cfg_.enable_whitening) return {};

            auto all_vecs = feature_bank_->ExtractAllVectors();
            auto num_samples = feature_bank_->NumSamples();
            auto embed_dim = cfg_.embed_dim;

            // 将扁平向量转换为 Embedding 对象供 FitPca 使用
            std::vector<sai::embedding::Embedding> samples;
            samples.reserve(num_samples);
            for (std::size_t i = 0; i < num_samples; ++i) {
                sai::embedding::EmbeddingMeta meta;
                meta.model_name = "coreset";
                meta.type = sai::embedding::EmbeddingType::Patch;
                meta.dim = embed_dim;
                meta.count = 1;
                meta.grid = {1, 1};
                std::vector<float> vec(all_vecs.begin() + static_cast<std::ptrdiff_t>(i * embed_dim),
                                        all_vecs.begin() + static_cast<std::ptrdiff_t>((i + 1) * embed_dim));
                samples.push_back(
                    sai::embedding::Embedding::FromCpu(std::move(vec), std::move(meta)));
            }

            auto total_k = embed_dim;
            auto fit_k = total_k + cfg_.drop_k;

            return sai::embedding::DimensionReducer::FitPca(samples, fit_k)
                .and_then([this, all_vecs = std::move(all_vecs), num_samples, embed_dim, total_k](
                              sai::embedding::DimensionReducer::PcaParams&& pca) -> Result<void> {
                    // 用第 [drop_k, drop_k+total_k) 个主成分手动构建 WhiteningParams
                    sai::embedding::DimensionReducer::WhiteningParams wp;
                    wp.target_dim = total_k;
                    wp.transform.resize(total_k * embed_dim, 0.0F);
                    for (std::size_t t = 0; t < total_k; ++t) {
                        float scale = 1.0F / std::sqrt(pca.eigvals[cfg_.drop_k + t] + 1e-6F);
                        for (std::size_t j = 0; j < embed_dim; ++j) {
                            wp.transform[t * embed_dim + j] =
                                pca.components[(cfg_.drop_k + t) * embed_dim + j] * scale;
                        }
                    }

                    // 白化 coreset 并重建索引
                    auto whitened = ApplyWhitening(all_vecs.data(), num_samples,
                                                    embed_dim, wp);
                    feature_bank_->Rebuild(whitened.data(), num_samples, total_k);
                    whitening_params_ = std::move(wp);
                    return {};
                });
        })
        .and_then([this]() -> Result<void> {
            if (!feature_bank_) return Result<void>{};
            if (cfg_.enable_adaptive_threshold) {
                float threshold_raw = ComputeAdaptiveThreshold(
                    *feature_bank_, cfg_.target_fpr, cfg_.k_nearest, ref_dist_);
                // Normalize threshold to ref_dist_ scale: OK patches ~ [0,1], NG patches > 1.
                effective_threshold_ = (ref_dist_ > 0.0F)
                    ? (threshold_raw / ref_dist_) : 1.0F;
            } else {
                effective_threshold_ = cfg_.anomaly_threshold;
                ref_dist_ = 1.0F;  // no normalization without adaptive threshold
            }
            return {};
        })
        .and_then([this]() -> Result<void> {
            if (cfg_.pca_model_path.empty()) return {};
            return sai::embedding::DimensionReducer::LoadPcaParams(cfg_.pca_model_path)
                .and_then([this](sai::embedding::DimensionReducer::PcaParams&& params) -> Result<void> {
                    pca_params_ = std::move(params);
                    return {};
                });
        });
}

// ── PatchCore::Detect ───────────────────────────────────────────────

auto PatchCore::Detect(const sai::embedding::Embedding& embedding) noexcept
    -> Result<DetectionResult> {
    if (!feature_bank_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PatchCore not initialized — FeatureBank not loaded",
            std::source_location::current(),
        });
    }

    const auto& meta = embedding.Meta();
    auto grid_h = meta.grid[0];
    auto grid_w = meta.grid[1];

    // 1. 验证 grid 与 config 匹配
    auto grid_ok = ValidatePatchGrid(grid_h, grid_w, cfg_.image_height, cfg_.image_width, cfg_.patch_size);
    if (!grid_ok.has_value()) {
        return tl::make_unexpected(grid_ok.error());
    }

    auto start = std::chrono::steady_clock::now();
    auto query_count = grid_h * grid_w;

#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    if (embedding.IsOnGpu() && !feature_bank_->IsOnGpu()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PatchCore: GPU embedding requires a GPU FeatureBank",
            std::source_location::current(),
        });
    }
#endif

    if (embedding.IsOnGpu()
        && (whitening_params_.has_value() || pca_params_.has_value())) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "PatchCore: GPU embedding cannot use CPU whitening or PCA scoring",
            std::source_location::current(),
        });
    }

    // 2. 保存原始查询指针（PCA 评分在原始向量上计算）
    const float* original_query = embedding.Data();
    const float* query_data = original_query;
    std::vector<float> whitened_query;

    // 3. 可选白化
    if (whitening_params_.has_value()) {
        whitened_query = ApplyWhitening(original_query, query_count,
                                         cfg_.embed_dim, *whitening_params_);
        query_data = whitened_query.data();
    }

    // 4. k-NN 搜索
    auto distances = feature_bank_->Search(query_data, query_count, cfg_.k_nearest);
    if (distances.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "k-NN search returned empty distances",
            std::source_location::current(),
        });
    }

    // 5. 提取 patch scores 并用全局参考值归一化。
    //    不使用 per-image max_dist 归一化——那会消除 OK/NG 图像的可区分性。
    //    ref_dist_ = coreset 自查询距离 max，在 Initialize() 中计算。
    //    OK patches: raw_dist ~ self_query_dist → score ~ [0, 1]
    //    NG patches: raw_dist >> self_query_dist → score > 1.0
    std::vector<float> patch_scores(query_count);
    if (ref_dist_ > 0.0F) {
        for (std::size_t i = 0; i < query_count; ++i)
            patch_scores[i] = distances[i * cfg_.k_nearest] / ref_dist_;
    } else {
        for (std::size_t i = 0; i < query_count; ++i)
            patch_scores[i] = distances[i * cfg_.k_nearest];
    }

    // 6. PCA 混合评分（E5）
    if (pca_params_.has_value()) {
        auto pca_scores = sai::embedding::DimensionReducer::Score(
            *pca_params_, original_query, query_count,
            cfg_.pca_score_method, cfg_.drop_k);

        float max_pca = *std::max_element(pca_scores.begin(), pca_scores.end());
        if (max_pca > 0.0F) {
            for (auto& s : pca_scores) s /= max_pca;
        }

        float alpha = cfg_.hybrid_alpha;
        for (std::size_t i = 0; i < query_count; ++i) {
            patch_scores[i] = alpha * patch_scores[i] + (1.0F - alpha) * pca_scores[i];
        }
    }

    // 7. 构建 DetectionResult（标准后处理管线）
    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto result = BuildDetectionResult(std::move(patch_scores), grid_h, grid_w,
                                        cfg_.image_height, cfg_.image_width,
                                        cfg_.gaussian_sigma,
                                        effective_threshold_,
                                        latency);

    // 8. 保存检测上下文（仅在 CoresetEvolution 启用时捕获）
    if (capture_context_) {
        std::shared_ptr<const std::vector<float>> embedding_data;
        if (embedding.IsOnGpu()) {
#if defined(SAI_CUDA_ENABLED)
            std::vector<float> host_embedding(query_count * cfg_.embed_dim);
            auto cuda_err = cudaMemcpy(
                host_embedding.data(), embedding.Data(), embedding.SizeBytes(),
                cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Runtime_GpuError,
                    std::string("PatchCore: context DtoH copy failed: ")
                        + cudaGetErrorString(cuda_err),
                    std::source_location::current(),
                });
            }
            embedding_data = std::make_shared<const std::vector<float>>(
                std::move(host_embedding));
#else
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Runtime_GpuError,
                "PatchCore: GPU context capture requires CUDA",
                std::source_location::current(),
            });
#endif
        } else {
            embedding_data = std::make_shared<const std::vector<float>>(
                embedding.Data(), embedding.Data() + query_count * cfg_.embed_dim);
        }
        last_ctx_.knn_distances = std::move(distances);
        last_ctx_.k_nearest = cfg_.k_nearest;
        last_ctx_.embedding_data = std::move(embedding_data);
        last_ctx_.grid_h = grid_h;
        last_ctx_.grid_w = grid_w;
        last_ctx_.dim = cfg_.embed_dim;
        last_ctx_.detection_result = result;
        last_ctx_.effective_threshold = effective_threshold_;
        last_ctx_.pca_image_score = 0.0F;
        last_ctx_.pca_self_query_p95 = 0.0F;
    }

    return result;
}

// ── PatchCore::DetectBatch ──────────────────────────────────────────

auto PatchCore::DetectBatch(
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>> {
    return DetectBatchImpl(*this, embeddings);
}

}  // namespace sai::detection
