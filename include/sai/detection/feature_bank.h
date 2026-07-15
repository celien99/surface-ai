// feature_bank.h — 批次 3.3 FeatureBank（FAISS k-NN 搜索后端）
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include <sai/core/error.h>
#include <sai/embedding/embedding.h>

namespace sai::retrieval { class VectorPath; }

// 前向声明——避免头文件泄漏 FAISS 内部类型给所有包含 feature_bank.h 的翻译单元
namespace faiss {
struct Index;
}  // namespace faiss

namespace sai::detection {

// FeatureBank：封装 FAISS IndexFlatL2（精确 L2 检索）的 coreset 特征库。
//
// 设计决策：
// - 使用 FAISS CPU 后端（IndexFlatL2）作为可移植路径，在 macOS arm64 / Linux x64 上均可构建和测试。
// - GPU 路径（feature_bank_cuda.cpp）通过 CMAKE 门控编译，仅在 CUDA SDK 可用时启用。
// - 构造为 move-only——FeatureBank 持有 FAISS Index 所有权，禁止拷贝。
// - LoadFromFile 读取原始 float32 little-endian 二进制文件（N×dim 矩阵），不依赖序列化格式。
class FeatureBank final {
public:
    // 从原始 float32 二进制文件加载 coreset。
    // path: N×dim float32 值的 raw 文件（little-endian，行主序）
    // dim: 每个向量的维度
    [[nodiscard]] static auto LoadFromFile(const std::filesystem::path& path,
                                           std::size_t dim) noexcept -> Result<FeatureBank>;

    // k-NN 搜索，返回距离矩阵（行主序，shape: query_count × k）。
    // query: query_count × dim 个 float32 值
    // query_count: 查询向量数量
    // k: 最近邻数量
    [[nodiscard]] auto Search(const float* query, std::size_t query_count,
                              std::size_t k) const noexcept -> std::vector<float>;

    [[nodiscard]] auto NumSamples() const noexcept -> std::size_t { return num_samples_; }
    [[nodiscard]] auto Dim() const noexcept -> std::size_t { return dim_; }

    // 提取所有向量为扁平 float 数组（N×D，行主序）
    [[nodiscard]] auto ExtractAllVectors() const noexcept -> std::vector<float>;

    // 用新向量集重建 FAISS 索引
    auto Rebuild(const float* vectors, std::size_t count, std::size_t dim) noexcept -> void;

    // 将 coreset 保存为原始 float32 二进制文件（LoadFromFile 的逆操作）。
    // path: N×dim float32 值（little-endian，行主序）
    [[nodiscard]] auto SaveToFile(const std::filesystem::path& path) const noexcept -> Result<void>;

    // 从多个 Embedding（正常样本）构建 coreset FeatureBank。
    // 提取所有 patch 向量，均匀采样至 max_samples 个，构建 FAISS 索引。
    // dim 必须与 Embedding 的 dim 一致。
    [[nodiscard]] static auto BuildFromEmbeddings(
        std::span<const sai::embedding::Embedding* const> embeddings,
        std::size_t dim,
        std::size_t max_samples = 10000) noexcept -> Result<FeatureBank>;

    // Greedy coreset selection via furthest-point sampling.
    // Extracts all patch vectors from embeddings, then iteratively selects
    // patches that maximize coverage of the normal manifold (minimize max
    // distance to nearest coreset point). Produces a more representative
    // coreset than uniform subsampling, at O(M·N·D) cost.
    [[nodiscard]] static auto BuildWithGreedyCoreset(
        std::span<const sai::embedding::Embedding* const> embeddings,
        std::size_t dim,
        std::size_t max_samples = 10000) noexcept -> Result<FeatureBank>;

    // GPU acceleration (only available when SAI_CUDA_ENABLED is defined).
#if defined(SAI_CUDA_ENABLED)
    [[nodiscard]] auto ToGpu(int device = 0) noexcept -> Result<void>;
    [[nodiscard]] auto IsOnGpu() const noexcept -> bool;
#endif

    // move-only
    FeatureBank(FeatureBank&&) noexcept;
    auto operator=(FeatureBank&&) noexcept -> FeatureBank&;
    FeatureBank(const FeatureBank&) = delete;
    auto operator=(const FeatureBank&) -> FeatureBank& = delete;

    ~FeatureBank();

private:
    friend class sai::retrieval::VectorPath;

    FeatureBank() noexcept;

    std::unique_ptr<faiss::Index> index_;
    std::size_t dim_ = 0;
    std::size_t num_samples_ = 0;

#if defined(SAI_CUDA_ENABLED)
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources_;
    std::unique_ptr<faiss::Index> gpu_index_;
    bool on_gpu_ = false;
#endif
};

}  // namespace sai::detection
