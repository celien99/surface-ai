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
#include <sai/device/device.h>

namespace sai::detection {

// ── PatchCore::Initialize ───────────────────────────────────────────

auto PatchCore::Initialize(sai::Context& /*ctx*/) noexcept -> Result<void> {
    auto bank = FeatureBank::LoadFromFile(cfg_.feature_bank_path, cfg_.embed_dim);
    if (!bank.has_value()) {
        return tl::make_unexpected(bank.error());
    }
    feature_bank_ = std::make_unique<FeatureBank>(std::move(bank.value()));
    return {};
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
    auto expected_grid_h = cfg_.image_height / cfg_.patch_size;
    auto expected_grid_w = cfg_.image_width / cfg_.patch_size;
    if (grid_h != expected_grid_h || grid_w != expected_grid_w) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_InvalidPatchGrid,
            "patch grid " + std::to_string(grid_h) + "×" + std::to_string(grid_w)
                + " does not match config " + std::to_string(expected_grid_h)
                + "×" + std::to_string(expected_grid_w),
            std::source_location::current(),
        });
    }

    auto start = std::chrono::steady_clock::now();

    // 2. k-NN 搜索
    auto query_count = grid_h * grid_w;
    auto distances = feature_bank_->Search(embedding.Data(), query_count, cfg_.k_nearest);
    if (distances.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "k-NN search returned empty distances",
            std::source_location::current(),
        });
    }

    // 取每个 query 的第一个最近邻距离（k=1 时 distances 即 1-per-query）
    // distances 是 nq × k 行主序；取每行第一个元素
    std::vector<float> patch_scores(query_count);
    for (std::size_t i = 0; i < query_count; ++i) {
        patch_scores[i] = distances[i * cfg_.k_nearest];
    }

    // 归一化到 [0,1]：除以最大距离（如果所有距离为 0，保留 0）
    float max_dist = *std::max_element(patch_scores.begin(), patch_scores.end());
    if (max_dist > 0.0F) {
        for (auto& s : patch_scores) s /= max_dist;
    }

    // 3. 构建 AnomalyMap
    AnomalyMap anomaly_map;
    anomaly_map.grid_h = grid_h;
    anomaly_map.grid_w = grid_w;
    anomaly_map.scores = std::move(patch_scores);

    auto image_level_score = anomaly_map.MaxScore();

    // 4. 双线性上采样到图像分辨率
    auto upsampled = BilinearUpsample(anomaly_map.scores.data(),
                                      grid_h, grid_w,
                                      cfg_.image_height, cfg_.image_width);

    // 5. Gaussian 平滑
    auto blurred = GaussianBlur(upsampled.data(), cfg_.image_height, cfg_.image_width,
                                cfg_.gaussian_sigma);

    // 6. 阈值 → 二值 mask → 连通分量
    std::vector<float> binary(cfg_.image_height * cfg_.image_width);
    for (std::size_t i = 0; i < binary.size(); ++i) {
        binary[i] = (blurred[i] > cfg_.anomaly_threshold) ? 1.0F : 0.0F;
    }
    auto regions = ConnectedComponents(binary.data(), cfg_.image_height, cfg_.image_width,
                                       blurred.data());

    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // 7. 构建 DetectionResult
    return DetectionResult{
        std::move(anomaly_map),
        std::move(regions),
        image_level_score,
        latency,
    };
}

// ── PatchCore::DetectBatch ──────────────────────────────────────────

auto PatchCore::DetectBatch(
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>> {
    std::vector<DetectionResult> results;
    results.reserve(embeddings.size());

    for (const auto* emb : embeddings) {
        if (emb == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_InvalidPatchGrid,
                "null embedding pointer in DetectBatch",
                std::source_location::current(),
            });
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
