// embedding.h — 批次 3.2 Embedding 双存储数据类型
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stop_token>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/memory/memory_pool.h>

namespace sai::memory {
class PinnedPool;
}  // namespace sai::memory

namespace sai::runtime {
class GpuStreamQueue;
template <typename T>
struct Task;
}  // namespace sai::runtime

namespace sai::embedding {

// Embedding 的来源类型：patch-level（DINOv3 的 dense features）vs global（CLIP 的 [CLS] token）。
enum class EmbeddingType : std::uint8_t { Patch, Global };

// Embedding 元数据——描述特征的来源、维度和形状。
// lifetime: 由 Embedding 持有，与 Embedding 等长。
struct EmbeddingMeta {
    std::string model_name;                         // 来源模型名称，如 "DINOv3" / "CLIP"
    EmbeddingType type = EmbeddingType::Patch;      // Patch（grid）或 Global（单向量）
    std::size_t dim = 0;                            // 每个向量的维度（如 DINOv3 1024, CLIP 512）
    std::size_t count = 0;                          // 向量个数（Patch = grid_h × grid_w, Global = 1）
    std::array<std::size_t, 2> grid{0, 0};          // patch grid 形状 {grid_h, grid_w}；Global 为 {0, 0}
    std::chrono::nanoseconds inference_latency{0};  // DINOv3/CLIP 推理耗时
};

// Embedding——双存储（GPU/CPU），move-only，标准化特征向量。
//
// 设计决策：
// - 双存储（GPU PooledPtr / CPU vector<float>），且仅有一个被填充：IsOnGpu() 区分。
// - move-only：拷贝 Embedding 会复制大量特征数据（~256MB for 256x256x1024 float32），禁止。
// - Data() 返回 const float*——GPU 路径返回显存指针，CPU 路径返回 vector 指针。
// - 生产者（Embedder）和消费者（Detector/FAISS/FeatureCache）通过 move 传递所有权，
//   零拷贝。
// - ToCpuAsync 声明但不定义（CUDA 门控，需要 GpuStreamQueue + PinnedPool）。
class Embedding final {
public:
    // 从 GPU 特征数据构造——device_data 持有 GpuPool 分配的显存 slab。
    [[nodiscard]] static auto FromGpu(sai::memory::PooledPtr<std::uint8_t> device_data,
                                       EmbeddingMeta meta) noexcept -> Embedding;

    // 从 CPU vector<float> 构造。
    [[nodiscard]] static auto FromCpu(std::vector<float> data,
                                       EmbeddingMeta meta) noexcept -> Embedding;

    // 返回指向存储数据的 float* 指针。
    // GPU 路径：返回 reinterpret_cast<const float*>(device_buffer_.Get())。
    // CPU 路径：返回 cpu_data_.data()。
    [[nodiscard]] auto Data() const noexcept -> const float*;

    // 元数据引用（生命周期与 Embedding 绑定）。
    [[nodiscard]] auto Meta() const noexcept -> const EmbeddingMeta& { return meta_; }

    // 总字节大小 = meta_.count * meta_.dim * sizeof(float)。
    [[nodiscard]] auto SizeBytes() const noexcept -> std::size_t;

    // true = 数据驻留在 GPU 显存（device_buffer_ 持有），false = 数据在 CPU 内存（cpu_data_）。
    [[nodiscard]] auto IsOnGpu() const noexcept -> bool { return on_gpu_; }

    // 声明但不定义（CUDA 门控）——需要 GpuStreamQueue + PinnedPool 实现异步 DtoH 搬移。
    // 定义位于 CUDA 门控的 .cpp 文件中，macOS 编译时不可用。
    [[nodiscard]] auto ToCpuAsync(sai::runtime::GpuStreamQueue& queue,
                                   sai::memory::PinnedPool& pinned,
                                   std::stop_token token) noexcept
        -> sai::runtime::Task<sai::Result<void>>;

    // move-only：拷贝会复制大量特征数据，禁止。
    Embedding(Embedding&&) noexcept = default;
    auto operator=(Embedding&&) noexcept -> Embedding& = default;
    Embedding(const Embedding&) = delete;
    auto operator=(const Embedding&) -> Embedding& = delete;

    ~Embedding() noexcept;

private:
    Embedding() noexcept = default;

    sai::memory::PooledPtr<std::uint8_t> device_buffer_{};
    std::vector<float> cpu_data_{};
    EmbeddingMeta meta_{};
    bool on_gpu_ = false;
};

}  // namespace sai::embedding
