// patch_core.h — 批次 3.3 PatchCore 检测器
#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <sai/detection/detector.h>
#include <sai/detection/feature_bank.h>
#include <sai/embedding/dimension_reducer.h>

namespace sai::detection {

// PatchCore：基于 coreset k-NN 的工业异常检测算法
//
// Pipeline: k-NN 搜索 → AnomalyMap → 上采样 → Gaussian 平滑 → 连通分量标记
//
// E3 文档引导——多层特征融合（使用 MultiLayerAggregator）的推荐配置：
//   上游在调用 PatchCore::Detect 之前，通过 MultiLayerAggregator 将 DINOv3
//   多层注意力特征（layer 3/6/9/12）聚合为单张 embedding，再送入 patch grid。
//   MultiLayerAggregator 配置示例（YAML）：
//     aggregator:
//       layers: [3, 6, 9, 12]
//       fusion: "concat"          # concat → 线性投影；或 "weighted_sum"
//       background_mask: true     # 启用 DINO 注意力图背景抑制
//   聚合后 embedding dim 应与 PatchCore::Config::embed_dim 对齐。
class PatchCore final : public IDetector {
public:
    struct Config {
        // === 现有字段（不变）===
        std::filesystem::path feature_bank_path;   // coreset 文件路径
        float anomaly_threshold = 0.8F;             // 缺陷判定阈值
        std::size_t k_nearest = 1;                  // k-NN 的 k 值
        std::size_t gaussian_sigma = 4;             // Gaussian 平滑 sigma（锚定 patch_size）
        std::size_t image_width = 518;               // 输入图像宽度
        std::size_t image_height = 518;              // 输入图像高度
        std::size_t patch_size = 14;                 // patch 边长（DINOv3 ViT 步长）
        std::size_t embed_dim = 1024;                // 特征维度（DINOv3 输出维度）

        // === E1+E2: PCA 白化 + 丢弃前 k 个主成分 ===
        bool enable_whitening = false;
        std::size_t drop_k = 0;          // 丢弃前 k 个主成分（白化和 PCA 评分共用）

        // === E4: 自适应阈值 ===
        bool enable_adaptive_threshold = false;
        float target_fpr = 0.01F;        // 目标误报率

        // === E5: k-NN × PCA 混合评分 ===
        std::filesystem::path pca_model_path;  // 空 = 禁用
        float hybrid_alpha = 0.5F;             // k-NN 权重
        sai::embedding::PcaScoreMethod pca_score_method =
            sai::embedding::PcaScoreMethod::Reconstruction;
    };

    explicit PatchCore(Config cfg) noexcept;

    [[nodiscard]] auto Initialize(sai::Context& ctx) noexcept -> Result<void> override;

    [[nodiscard]] auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> override;

    [[nodiscard]] auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> override;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "PatchCore"; }

    // 注入已加载的 FeatureBank（替代从 feature_bank_path 加载）。
    // 调用方负责确保 dim 与 embed_dim 匹配。在 Initialize() 之前调用。
    auto SetFeatureBank(std::unique_ptr<FeatureBank> fb) noexcept -> void {
        feature_bank_ = std::move(fb);
    }

    // 原子替换 FeatureBank，返回旧 bank。
    // 用于 CoresetEvolution 的 double-buffer swap——旧 bank 由调用方回收作为 standby。
    // 可在运行时调用（检测线程正在读取旧 bank → 旧 bank 由返回的 unique_ptr 保持存活）。
    [[nodiscard]] auto SwapFeatureBank(std::unique_ptr<FeatureBank> new_bank) noexcept
        -> std::unique_ptr<FeatureBank> {
        auto old = std::move(feature_bank_);
        feature_bank_ = std::move(new_bank);
        return old;
    }

    // ── 最近一帧检测上下文（供 CoresetEvolution 访问） ──
    // 每次 Detect() 调用后更新。调用方在 ResultCallback 中使用。
    struct DetectionContext {
        std::vector<float> knn_distances;      // query_count × k_nearest
        std::size_t k_nearest = 0;
        std::vector<float> embedding_data;     // query_count × dim
        std::size_t grid_h = 0;
        std::size_t grid_w = 0;
        std::size_t dim = 0;
        DetectionResult detection_result;      // 最近一帧检测结果
        float effective_threshold = 0.0F;
        float pca_image_score = 0.0F;          // 0.0 = PCA 未启用
        float pca_self_query_p95 = 0.0F;       // 0.0 = PCA 未启用
    };

    [[nodiscard]] auto LastContext() const noexcept -> const DetectionContext& {
        return last_ctx_;
    }

    // Object（基类）禁止移动/拷贝，故 PatchCore 继承此约束
    PatchCore(PatchCore&&) noexcept = delete;
    PatchCore(const PatchCore&) = delete;

private:
    Config cfg_;
    std::unique_ptr<FeatureBank> feature_bank_;

    // 最近一帧检测上下文（每次 Detect 后更新）
    DetectionContext last_ctx_;

    // E1+E2: 白化参数
    std::optional<sai::embedding::DimensionReducer::WhiteningParams> whitening_params_;

    // E4: 自检计算的阈值
    float effective_threshold_ = 0.8F;

    // E5: 已加载的 PCA 模型
    std::optional<sai::embedding::DimensionReducer::PcaParams> pca_params_;
};

inline PatchCore::PatchCore(Config cfg) noexcept
    : cfg_(std::move(cfg)) {}

}  // namespace sai::detection
