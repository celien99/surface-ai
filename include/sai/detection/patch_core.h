// patch_core.h — 批次 3.3 PatchCore 检测器（声明 + Config，无 Detect 实现体）
#pragma once

#include <filesystem>
#include <string_view>

#include <sai/detection/detector.h>

namespace sai::detection {

// 前向声明——FeatureBank 在 Task 8 的 feature_bank.h 中完整定义
class FeatureBank;

// PatchCore：基于 coreset k-NN 的工业异常检测算法
//
// Pipeline: k-NN 搜索 → AnomalyMap → 上采样 → Gaussian 平滑 → 连通分量标记
class PatchCore final : public IDetector {
public:
    struct Config {
        std::filesystem::path feature_bank_path;   // coreset 文件路径
        float anomaly_threshold = 0.8F;             // 缺陷判定阈值
        std::size_t k_nearest = 1;                  // k-NN 的 k 值
        std::size_t gaussian_sigma = 4;             // Gaussian 平滑 sigma（锚定 patch_size）
        std::size_t image_width = 518;               // 输入图像宽度
        std::size_t image_height = 518;              // 输入图像高度
        std::size_t patch_size = 14;                 // patch 边长（DINOv3 ViT 步长）
        std::size_t embed_dim = 1024;                // 特征维度（DINOv3 输出维度）
    };

    explicit PatchCore(Config cfg) noexcept;

    [[nodiscard]] auto Initialize(sai::Context& ctx) noexcept -> Result<void> override;

    // 声明但不定义（Task 8 实现）——Task 7 仅提供接口骨架
    [[nodiscard]] auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> override;

    // 声明但不定义（Task 8 实现）
    [[nodiscard]] auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> override;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "PatchCore"; }

    // Object（基类）禁止移动/拷贝，故 PatchCore 继承此约束
    PatchCore(PatchCore&&) noexcept = delete;
    PatchCore(const PatchCore&) = delete;

private:
    Config cfg_;
    // Task 7 使用指针（FeatureBank 仅前向声明），Task 8 切回值语义
    FeatureBank* feature_bank_ = nullptr;
};

inline PatchCore::PatchCore(Config cfg) noexcept
    : cfg_(std::move(cfg)) {}

}  // namespace sai::detection
