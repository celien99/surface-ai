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
    return sai::embedding::DimensionReducer::LoadPcaParams(cfg_.pca_model_path)
        .and_then([this](sai::embedding::DimensionReducer::PcaParams&& params) -> Result<void> {
            auto loaded_dim = params.mean.size();
            if (loaded_dim != cfg_.embed_dim) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Detection_FeatureBankLoadFailed,
                    "PcaDetector: PCA model dim (" + std::to_string(loaded_dim)
                        + ") != config embed_dim (" + std::to_string(cfg_.embed_dim) + ")",
                    std::source_location::current()});
            }
            pca_params_ = std::move(params);
            initialized_ = true;
            return {};
        });
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
    return ValidatePatchGrid(grid_h, grid_w, cfg_.image_height, cfg_.image_width, cfg_.patch_size)
        .and_then([&]() -> Result<DetectionResult> {
            auto start = std::chrono::steady_clock::now();

            auto query_count = grid_h * grid_w;

            // 2. PCA 评分
            auto scores = sai::embedding::DimensionReducer::Score(
                pca_params_, embedding.Data(), query_count,
                cfg_.score_method, cfg_.drop_k);

            if (scores.empty()) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Detection_FeatureBankLoadFailed,
                    "PCA scoring returned empty scores",
                    std::source_location::current()});
            }

            // 3. 归一化到 [0,1]
            float max_score = *std::max_element(scores.begin(), scores.end());
            if (max_score > 0.0F) {
                for (auto& s : scores) s /= max_score;
            }

            auto end = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

            // 4. 构建 DetectionResult（标准后处理管线）
            return BuildDetectionResult(std::move(scores), grid_h, grid_w,
                                        cfg_.image_height, cfg_.image_width,
                                        cfg_.gaussian_sigma,
                                        cfg_.anomaly_threshold,
                                        latency);
        });
}

// ── DetectBatch ─────────────────────────────────────────────────────

auto PcaDetector::DetectBatch(
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>> {
    return DetectBatchImpl(*this, embeddings);
}

}  // namespace sai::detection
