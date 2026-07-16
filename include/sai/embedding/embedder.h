// embedder.h — IEmbedder 接口与 PatchEmbedder / GlobalEmbedder（批次 3.2）
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/embedding/embedding.h>
#include <sai/image/image.h>
#include <sai/inference/clip_adapter.h>
#include <sai/inference/dino_v3_adapter.h>

namespace sai::embedding {

// IEmbedder——统一特征提取接口。不派生自 IService/IModule；
// Embedder 是算法组件，由 Detector 或 Pipeline 直接持有。
// 输入为基类 Image 引用，实现层负责校验 GPU 存储类型。
class IEmbedder : public sai::Object {
public:
    [[nodiscard]] virtual auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> = 0;

    [[nodiscard]] virtual auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> = 0;

    [[nodiscard]] virtual auto ModelName() const noexcept -> std::string_view = 0;
};

// PatchEmbedder — DINOv3 adapter → patch feature grid → Embedding。
// move-only，持有 DinoV3Adapter。Extract 仅在输入为 GpuImage 时有效。
class PatchEmbedder final : public IEmbedder {
public:
    [[nodiscard]] static auto Create(sai::inference::DinoV3Adapter adapter) noexcept
        -> Result<PatchEmbedder>;

    [[nodiscard]] auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> override;
    [[nodiscard]] auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "DINOv3"; }

    PatchEmbedder(PatchEmbedder&&) noexcept;
    auto operator=(PatchEmbedder&&) noexcept -> PatchEmbedder&;
    ~PatchEmbedder();
    PatchEmbedder(const PatchEmbedder&) = delete;
    auto operator=(const PatchEmbedder&) -> PatchEmbedder& = delete;

    // GPU inference path — defined in patch_embedder_cuda.cpp (CUDA-gated).
    // On non-CUDA builds, a stub returning an error is provided in patch_embedder.cpp.
    [[nodiscard]] auto ExtractGpu(const sai::image::Image& image) noexcept
        -> Result<Embedding>;

    // Inject a GpuPool for zero-copy GPU feature extraction.
    // When set, ExtractGpu copies TRT output → pool buffer (D2D) and
    // returns Embedding::FromGpu (zero host round-trip).
    // When nullptr (default), falls back to D2H copy → Embedding::FromCpu.
    auto SetGpuPool(sai::memory::IMemoryPool* pool) noexcept -> void {
        gpu_pool_ = pool;
    }

private:
    explicit PatchEmbedder(sai::inference::DinoV3Adapter adapter) noexcept;
    sai::inference::DinoV3Adapter adapter_;
    sai::memory::IMemoryPool* gpu_pool_ = nullptr;
    void* cuda_stream_ = nullptr;  // 每实例独立 CUDA stream，避免默认流串行化
    bool has_adapter_ = true;
};

// GlobalEmbedder — CLIP adapter → [CLS] token → Embedding。
// move-only，持有 ClipAdapter。Extract 仅在输入为 GpuImage 时有效。
class GlobalEmbedder final : public IEmbedder {
public:
    [[nodiscard]] static auto Create(sai::inference::ClipAdapter adapter) noexcept
        -> Result<GlobalEmbedder>;

    [[nodiscard]] auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> override;
    [[nodiscard]] auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override { return "CLIP"; }

    GlobalEmbedder(GlobalEmbedder&&) noexcept;
    auto operator=(GlobalEmbedder&&) noexcept -> GlobalEmbedder&;
    ~GlobalEmbedder();
    GlobalEmbedder(const GlobalEmbedder&) = delete;
    auto operator=(const GlobalEmbedder&) -> GlobalEmbedder& = delete;

    // GPU inference path — defined in global_embedder_cuda.cpp (CUDA-gated).
    [[nodiscard]] auto ExtractGpu(const sai::image::Image& image) noexcept
        -> Result<Embedding>;

private:
    explicit GlobalEmbedder(sai::inference::ClipAdapter adapter) noexcept;
    sai::inference::ClipAdapter adapter_;
    void* cuda_stream_ = nullptr;  // 每实例独立 CUDA stream，避免默认流串行化
    bool has_adapter_ = true;
};

}  // namespace sai::embedding
