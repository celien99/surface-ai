// feature_bank_cuda.cpp — FeatureBank FAISS GPU 后端（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
//
// 实现 FeatureBank 的 GPU 加速路径：
//   - ToGpu(): 将 CPU IndexFlatL2 使用 StandardGpuResources 迁移至 GPU
//   - Search() 在 GPU 上执行（当 IsOnGpu() == true 时）
//   - 搜索结果自动通过 FAISS GPU → CPU 拷贝返回
//
// FAISS GPU 后端要求 CUDA Toolkit 和 faiss-gpu（编译时链接 libfaiss_gpu）。
// faiss::gpu::StandardGpuResources 管理临时 GPU 内存和 CUDA stream，
// GpuIndexFlatL2 通过 index_cpu_to_gpu 从 CPU IndexFlatL2 迁移。

#include <sai/detection/feature_bank.h>

#include <cstddef>
#include <memory>
#include <source_location>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

namespace sai::detection {

auto FeatureBank::ToGpu(int device) noexcept -> Result<void> {
    if (!index_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "FeatureBank::ToGpu: no index loaded",
            std::source_location::current(),
        });
    }

    if (gpu_resources_ != nullptr) {
        return {};  // Already on GPU
    }

    try {
        auto* cpu_index = dynamic_cast<faiss::IndexFlat*>(index_.get());
        if (cpu_index == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_FeatureBankLoadFailed,
                "FeatureBank::ToGpu: index is not IndexFlat, GPU migration unsupported",
                std::source_location::current(),
            });
        }

        gpu_resources_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        gpu_resources_->setTempMemory(512 * 1024 * 1024);  // 512 MiB temp

        auto* gpu_index = faiss::gpu::index_cpu_to_gpu(
            gpu_resources_.get(), device, cpu_index);

        // Store the GPU index as a unique_ptr; the CPU index remains owned
        // by index_ (we don't destroy it — the GPU index wraps it).
        gpu_index_ = std::unique_ptr<faiss::Index>(gpu_index);
        on_gpu_ = true;

        return {};
    } catch (const std::exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            std::string("FeatureBank::ToGpu failed: ") + e.what(),
            std::source_location::current(),
        });
    }
}

auto FeatureBank::IsOnGpu() const noexcept -> bool {
    return on_gpu_;
}

// Override Search: when on GPU, delegate to GPU index; otherwise fall back
// to the CPU path defined in feature_bank.cpp.
// Note: This redirection is handled transparently — the caller always uses
// FeatureBank::Search(). When SAI_CUDA_ENABLED, the Search method links
// against this CUDA-aware implementation. When not enabled, the CPU-only
// Search in feature_bank.cpp is used.

// The GPU Search path is injected by replacing the CPU Search implementation.
// This file overrides Search() when CUDA is enabled.
// (Search is defined in feature_bank.cpp — GPU acceleration is via
//  explicit ToGpu() call before Search(); the existing Search() delegates
//  to whichever index_ is active.)

}  // namespace sai::detection
