// pca_detector.cpp — 基于 PCA 子空间建模的异常检测器实现
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <source_location>
#include <vector>

#include <sai/detection/post_process_utils.h>
#include <sai/detection/pca_detector.h>
#include <sai/embedding/dimension_reducer.h>

namespace sai::detection {

// ── Initialize ──────────────────────────────────────────────────────

auto PcaDetector::Initialize(sai::Context& /*ctx*/) noexcept -> Result<void> {
    auto params = sai::embedding::DimensionReducer::LoadPcaParams(cfg_.pca_model_path);
    if (!params.has_value()) {
        return tl::make_unexpected(params.error());
    }

    // 验证维度匹配
    auto loaded_dim = params->mean.size();
    if (loaded_dim != cfg_.embed_dim) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PcaDetector: PCA model dim (" + std::to_string(loaded_dim)
                + ") != config embed_dim (" + std::to_string(cfg_.embed_dim) + ")",
            std::source_location::current()});
    }

    pca_params_ = std::move(params.value());
    initialized_ = true;
    return {};
}

// ── Detect ──────────────────────────────────────────────────────────

auto PcaDetector::Detect(const sai::embedding::Embedding& embedding) noexcept
    -> Result<DetectionResult> {
    if (!initialized_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PcaDetector not initialized — PCA model not loaded",
            std::source_location::current()});
    }

    const auto& meta = embedding.Meta();
    auto grid_h = meta.grid[0];
    auto grid_w = meta.grid[1];

    // 1. 验证 grid 与 config 匹配
    auto expected_grid_h = cfg_.image_height / cfg_.patch_size;
    auto expected_grid_w = cfg_.image_width / cfg_.patch_size;
    if (grid_h != expected_grid_h || grid_w != expected_grid_w) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_InvalidPatchGrid,
            "patch grid " + std::to_string(grid_h) + "×" + std::to_string(grid_w)
                + " does not match config " + std::to_string(expected_grid_h)
                + "×" + std::to_string(expected_grid_w),
            std::source_location::current()});
    }

    auto start = std::chrono::steady_clock::now();

    // 2. PCA 评分
    auto query_count = grid_h * grid_w;
    auto scores = sai::embedding::DimensionReducer::Score(
        pca_params_, embedding.Data(), query_count,
        cfg_.score_method, cfg_.drop_k);

    if (scores.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PCA scoring returned empty scores",
            std::source_location::current()});
    }

    // 3. 图像级分数（归一化前）
    float raw_max = *std::max_element(scores.begin(), scores.end());
    float image_level_score = raw_max;

    // 4. 归一化到 [0,1]（仅用于 anomaly_map 和后处理）
    if (raw_max > 0.0F) {
        for (auto& s : scores) s /= raw_max;
    }

    // 5. 构建 AnomalyMap
    AnomalyMap anomaly_map;
    anomaly_map.grid_h = grid_h;
    anomaly_map.grid_w = grid_w;
    anomaly_map.scores = std::move(scores);

    // 6. 双线性上采样到图像分辨率
    auto upsampled = BilinearUpsample(anomaly_map.scores.data(),
                                      grid_h, grid_w,
                                      cfg_.image_height, cfg_.image_width);

    // 7. Gaussian 平滑
    auto blurred = GaussianBlur(upsampled.data(), cfg_.image_height, cfg_.image_width,
                                cfg_.gaussian_sigma);

    // 8. 阈值 → 二值 mask → 连通分量
    std::vector<float> binary(cfg_.image_height * cfg_.image_width);
    for (std::size_t i = 0; i < binary.size(); ++i) {
        binary[i] = (blurred[i] > cfg_.anomaly_threshold) ? 1.0F : 0.0F;
    }
    auto regions = ConnectedComponents(binary.data(), cfg_.image_height, cfg_.image_width,
                                       blurred.data());

    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // 8. 构建 DetectionResult
    return DetectionResult{
        std::move(anomaly_map),
        std::move(regions),
        image_level_score,
        latency,
    };
}

// ── DetectBatch ─────────────────────────────────────────────────────

auto PcaDetector::DetectBatch(
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>> {
    std::vector<DetectionResult> results;
    results.reserve(embeddings.size());

    for (const auto* emb : embeddings) {
        if (emb == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_InvalidPatchGrid,
                "null embedding pointer in DetectBatch",
                std::source_location::current()});
        }
        auto result = Detect(*emb);
        if (!result.has_value()) {
            return tl::make_unexpected(result.error());
        }
        results.push_back(std::move(result.value()));
    }
    return results;
}

}  // namespace sai::detection
