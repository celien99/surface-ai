// dimension_reducer.h — 批次 3.2 DimensionReducer PCA/Whitening 降维、评分、序列化与 Pooling
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include <sai/core/error.h>
#include <sai/embedding/embedding.h>

namespace sai::embedding {

// Pooling 策略——在 patch grid 上进行空间降采样
enum class PoolingStrategy : std::uint8_t { Average, Max };

// PCA 异常评分方法
enum class PcaScoreMethod : std::uint8_t {
    Reconstruction,  // ‖X - X_recon‖²
    Mahalanobis,     // Z · diag(1/λ) · Zᵀ
    Cosine,          // 1 - cos(X, X_recon)
    Euclidean,       // ‖Z‖²（主成分空间距离）
};

// DimensionReducer——PCA 降维、Whitening 球化、空间 Pooling，全部 CPU 端执行。
//
// 设计决策：
// - FitPca/FitWhitening 为静态方法，输出参数对象（PcaParams/WhiteningParams），
//   运行期不重新拟合。参数通常从标定文件加载。
// - Reduce 执行矩阵乘法降维，输入必须是 CPU Embedding（IsOnGpu() == false），
//   检查不通过返回 Embedding_DimensionMismatch。
// - Pool 空间池化：输入 Patch Embedding（grid_h × grid_w × dim）→ Global
//   Embedding（1 × dim），使用 Average 或 Max 策略。
class DimensionReducer final {
public:
    // PCA 参数——components 为 target_dim × input_dim 矩阵（行优先），
    // mean 为 input_dim 维的均值向量，eigvals 为 target_dim 个特征值（降序）。
    struct PcaParams {
        std::vector<float> components;
        std::size_t target_dim;
        std::vector<float> mean;
        std::vector<float> eigvals;   // 特征值，用于 Mahalanobis 评分和方差解释率计算
    };

    // Whitening 参数——transform 为 target_dim × input_dim 矩阵（行优先），
    // 已包含 PCA 旋转 + 特征值归一化。不做均值中心化。
    struct WhiteningParams {
        std::vector<float> transform;
        std::size_t target_dim;
    };

    // 静态拟合：对 samples 中所有 Embedding 的所有向量拟合 PCA，输出 target_dim 维。
    // samples 中的所有向量必须为同一维度，且 target_dim ≤ input_dim。
    // 返回 Embedding_DimensionMismatch 如果 target_dim > input_dim。
    [[nodiscard]] static auto FitPca(const std::vector<Embedding>& samples,
                                      std::size_t target_dim) noexcept -> Result<PcaParams>;

    // 静态拟合：PCA 球化变换，输出 target_dim 维——各向同性方差（≈1）。
    [[nodiscard]] static auto FitWhitening(const std::vector<Embedding>& samples,
                                            std::size_t target_dim) noexcept -> Result<WhiteningParams>;

    // ── PCA 异常评分 ──
    // 对 N 个 D 维特征向量 X（扁平数组，行主序）计算异常分数。
    // drop_k: 丢弃前 k 个主成分（编码主导方差→全局光照/纹理），丢弃后对小缺陷更敏感。
    [[nodiscard]] static auto Score(const PcaParams& params, const float* X,
                                     std::size_t N, PcaScoreMethod method,
                                     std::size_t drop_k = 0) noexcept -> std::vector<float>;

    // ── PcaParams 序列化 ──
    // 二进制格式（little-endian）：[D:u64][k:u64][mu: D×f32][eigvals: k×f32][components: k×D×f32]
    [[nodiscard]] static auto SavePcaParams(const PcaParams& params,
                                             const std::filesystem::path& path) noexcept -> Result<void>;
    [[nodiscard]] static auto LoadPcaParams(const std::filesystem::path& path) noexcept -> Result<PcaParams>;

    // ── 流式 PCA 拟合 ──
    // 两遍流式算法（GPU 友好），处理超出内存的大数据集。
    // make_generator: 返回一个新的 batch generator，每次调用从头开始产生 batch。
    //   batch generator: 每次调用返回一批特征向量（扁平 float 数组，D 维 × batch_N 个），
    //                    返回空 vector 表示数据结束。
    // total_N: 总特征向量数（用于协方差归一化）
    // target_k: 保留的主成分数（0 = 保留全部，即 target_k = D）
    [[nodiscard]] static auto FitPcaStreaming(
        std::function<std::function<std::vector<float>()>()> make_generator,
        std::size_t D, std::size_t total_N, std::size_t target_k) noexcept -> Result<PcaParams>;

    // 用已拟合的 PCA 参数构造。
    explicit DimensionReducer(PcaParams params) noexcept;

    // 用已拟合的 Whitening 参数构造。
    explicit DimensionReducer(WhiteningParams params) noexcept;

    // 对单个 Embedding 降维：input 的 dim 必须匹配 params 的 input_dim。
    // 返回降维后的 Embedding（count 不变，dim = target_dim）。
    [[nodiscard]] auto Reduce(const Embedding& input) noexcept -> Result<Embedding>;

    // 批量降维——每 Embedding 独立降维，单个失败不影响其他。
    [[nodiscard]] auto ReduceBatch(const std::vector<Embedding>& inputs) noexcept
        -> Result<std::vector<Embedding>>;

    // 空间 Pooling——将 patch grid（grid_h × grid_w × dim）池化为单向量（1 × dim）。
    // 非 grid Embedding（Global 类型或 count=1）视为已池化，直接返回副本。
    [[nodiscard]] static auto Pool(const Embedding& input,
                                    PoolingStrategy strategy) noexcept -> Result<Embedding>;

    DimensionReducer(const DimensionReducer&) = delete;
    auto operator=(const DimensionReducer&) -> DimensionReducer& = delete;
    DimensionReducer(DimensionReducer&&) noexcept = default;
    auto operator=(DimensionReducer&&) noexcept -> DimensionReducer& = default;

private:
    bool is_pca_;                       // true = PCA（需要减均值），false = Whitening（无均值）
    std::vector<float> components_;     // target_dim × input_dim 矩阵（行优先）
    std::size_t target_dim_;
    std::size_t input_dim_;
    std::vector<float> mean_;           // input_dim 均值向量（仅 PCA 使用）

    // 内部 Helper：对 count × input_dim 的扁平数据应用降维 → count × target_dim
    [[nodiscard]] auto ApplyReduce(const float* data, std::size_t count) const noexcept
        -> Result<std::vector<float>>;
};

}  // namespace sai::embedding
