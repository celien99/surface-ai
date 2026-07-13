// pca_detector.h — 基于 PCA 子空间建模的异常检测器
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>

#include <sai/detection/detector.h>
#include <sai/embedding/dimension_reducer.h>

namespace sai::detection {

// PcaDetector——将 PCA 学习到的正常样本低维线性子空间作为检测依据。
//
// 与 PatchCore（k-NN + FAISS memory bank）互补：
//   - 存储量固定为 O(D×k)，不随训练样本数增长
//   - 推理为矩阵乘法 O(D×k)，延迟确定
//   - 低方差方向被天然丢弃，等价于隐式噪声正则化
//
// Pipeline: PCA 评分 → AnomalyMap → 上采样 → Gaussian 平滑 → 连通分量标记。
// 后处理步骤（上采样/平滑/连通域）与 PatchCore 共享 post_process_utils。
class PcaDetector final : public IDetector {
public:
    struct Config {
        std::filesystem::path pca_model_path;                         // 预拟合 PCA 参数文件
        sai::embedding::PcaScoreMethod score_method =
            sai::embedding::PcaScoreMethod::Reconstruction;          // 异常评分方法
        std::size_t drop_k = 0;                                        // 丢弃前 k 个主成分
        float anomaly_threshold = 0.8F;                                // 缺陷判定阈值
        std::size_t gaussian_sigma = 4;                                // Gaussian 平滑 sigma
        std::size_t image_width = 518;                                 // 输入图像宽度
        std::size_t image_height = 518;                                // 输入图像高度
        std::size_t patch_size = 14;                                   // patch 边长
        std::size_t embed_dim = 1024;                                  // 特征维度
    };

    explicit PcaDetector(Config cfg) noexcept;

    [[nodiscard]] auto Initialize(sai::Context& ctx) noexcept -> Result<void> override;

    [[nodiscard]] auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> override;

    [[nodiscard]] auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> override;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "PcaDetector"; }

    PcaDetector(PcaDetector&&) noexcept = delete;
    PcaDetector(const PcaDetector&) = delete;

private:
    Config cfg_;
    sai::embedding::DimensionReducer::PcaParams pca_params_;
    bool initialized_ = false;
};

inline PcaDetector::PcaDetector(Config cfg) noexcept
    : cfg_(std::move(cfg)) {}

}  // namespace sai::detection
