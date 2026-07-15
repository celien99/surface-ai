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

// Include runtime/task.h for the Task<T> alias (used in ToCpuAsync return type).
// The previous forward-declaration template<typename T> struct Task was incorrect —
// Task<T> is a coroutine_handle alias, not a class template.
#include <sai/runtime/task.h>

namespace sai::runtime {
class GpuStreamQueue;
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

    // Global features (CLIP image-level embedding, optional).
    // Set by InferenceStage when a GlobalEmbedder is configured.
    // Consumers (RuleEvalStage) use this for cross-modal vector retrieval.
    [[nodiscard]] auto GlobalFeatures() const noexcept -> const std::vector<float>&
    { return global_features_; }
    [[nodiscard]] auto HasGlobalFeatures() const noexcept -> bool
    { return !global_features_.empty(); }
    auto SetGlobalFeatures(std::vector<float> features) noexcept -> void
    { global_features_ = std::move(features); }

    // Surface identity carried from image metadata through the pipeline.
    [[nodiscard]] auto SurfaceId() const noexcept -> const std::string&
    { return surface_id_; }
    auto SetSurfaceId(std::string id) noexcept -> void
    { surface_id_ = std::move(id); }

    // Position identity for multi-position detection routing.
    [[nodiscard]] auto PositionId() const noexcept -> std::uint16_t
    { return position_id_; }
    auto SetPositionId(std::uint16_t id) noexcept -> void
    { position_id_ = std::move(id); }

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
    std::vector<float> global_features_{};
    std::string surface_id_{};
    std::uint16_t position_id_ = 0;
};

}  // namespace sai::embedding
