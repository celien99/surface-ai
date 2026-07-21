// dino_v3_adapter.cpp — DINO ViT Adapter 推理实现（批次 3.1，CUDA 门控）
// 该文件仅在目标平台上编译。Infer() 方法调用 TensorRtEngine 执行实际的 GPU 推理。
// DINOv2 输出包含 CLS token，需跳过首 token 再交 PatchEmbedder。

#include <sai/inference/dino_v3_adapter.h>

#include <sai/image/gpu_image.h>

#include <cuda_runtime.h>

#include <string>

namespace sai::inference {

namespace {

// Lazily allocate the GPU output buffer for "last_hidden_state" if it hasn't
// been allocated yet.  The pointer is stored in the adapter's output_buffer_
// and freed by the destructor / move-assignment.
[[nodiscard]] auto EnsureOutputBuffer(IInferenceEngine& engine,
                                       std::size_t output_floats,
                                       void*& output_buffer) noexcept -> Result<void> {
    const auto& outputs = engine.OutputBindings();
    const TensorBinding* binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            binding = &b;
            break;
        }
    }
    if (binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' not found",
        });
    }

    // Already allocated — reuse.
    if (binding->device_ptr != nullptr) {
        return {};
    }

    // Allocate GPU memory for the output tensor.
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, output_floats * sizeof(float));
    if (err != cudaSuccess) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            std::string("DinoV3Adapter: cudaMalloc for output buffer failed: ")
                + cudaGetErrorString(err),
        });
    }

    auto set_result = engine.SetTensorAddress("last_hidden_state", ptr);
    if (!set_result.has_value()) {
        cudaFree(ptr);
        return tl::make_unexpected(set_result.error());
    }

    output_buffer = ptr;  // transfer ownership to adapter for cleanup
    return {};
}

}  // namespace

// ── Lifetime management (defined here because cudaFree requires cuda_runtime.h) ──

DinoV3Adapter::~DinoV3Adapter() {
    if (output_buffer_ != nullptr) {
        cudaFree(output_buffer_);
    }
}

DinoV3Adapter::DinoV3Adapter(DinoV3Adapter&& other) noexcept
    : engine_(other.engine_),
      cfg_(std::move(other.cfg_)),
      output_buffer_(other.output_buffer_) {
    other.output_buffer_ = nullptr;
}

auto DinoV3Adapter::operator=(DinoV3Adapter&& other) noexcept -> DinoV3Adapter& {
    if (this != &other) {
        if (output_buffer_ != nullptr) {
            cudaFree(output_buffer_);
        }
        engine_ = other.engine_;
        cfg_ = std::move(other.cfg_);
        output_buffer_ = other.output_buffer_;
        other.output_buffer_ = nullptr;
    }
    return *this;
}

// ── Inference ──

auto DinoV3Adapter::Infer(const sai::image::GpuImage& image) noexcept -> Result<PatchFeatures> {
    // 1. 设置输入 tensor 的 GPU 地址——image.Data() 返回 const uint8_t*，
    //    但 SetTensorAddress 需要 void*（设备指针，const 对设备端无意义）。
    auto set_result = engine_->SetTensorAddress(
        "pixel_values",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. 确保输出 GPU buffer 已分配（首次调用懒分配，后续复用）。
    auto grid_h = cfg_.image_size / cfg_.patch_size;
    auto grid_w = cfg_.image_size / cfg_.patch_size;
    std::size_t output_floats = (grid_h * grid_w + 1) * cfg_.embed_dim;  // +1 for CLS
    auto buf_result = EnsureOutputBuffer(*engine_, output_floats, output_buffer_);
    if (!buf_result.has_value()) {
        return tl::make_unexpected(buf_result.error());
    }

    // 3. 执行同步 GPU 推理。
    auto infer_result = engine_->Infer();
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 4. 读取输出 binding，获取 GPU 端的 patch features。
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' not found",
        });
    }

    // 5. 校验输出大小。
    std::size_t actual_elements = features_binding->size_bytes / sizeof(float);
    if (actual_elements < output_floats) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: output size mismatch (expected " +
                       std::to_string(output_floats) + " floats, got " +
                       std::to_string(actual_elements) + ")",
        });
    }

    // 6. 跳过 CLS token（首个 token），返回纯 patch features
    return PatchFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr) + cfg_.embed_dim,
        .grid_h = grid_h,
        .grid_w = grid_w,
        .dim = cfg_.embed_dim,
    };
}

auto DinoV3Adapter::InferAsync(const sai::image::GpuImage& image,
                                void* stream) noexcept -> Result<PatchFeatures> {
    // 1. Set input tensor address (same as sync path)
    auto set_result = engine_->SetTensorAddress(
        "pixel_values",
        const_cast<std::uint8_t*>(image.Data()));
    if (!set_result.has_value()) {
        return tl::make_unexpected(set_result.error());
    }

    // 2. Ensure output GPU buffer is allocated (lazy, first-call allocation).
    auto grid_h = cfg_.image_size / cfg_.patch_size;
    auto grid_w = cfg_.image_size / cfg_.patch_size;
    std::size_t output_floats = (grid_h * grid_w + 1) * cfg_.embed_dim;  // +1 for CLS
    auto buf_result = EnsureOutputBuffer(*engine_, output_floats, output_buffer_);
    if (!buf_result.has_value()) {
        return tl::make_unexpected(buf_result.error());
    }

    // 3. Execute async GPU inference — no cudaStreamSynchronize.
    // The caller owns stream lifecycle and synchronization.
    auto infer_result = engine_->InferAsync(stream);
    if (!infer_result.has_value()) {
        return tl::make_unexpected(infer_result.error());
    }

    // 4. Read output binding (same as sync path)
    const auto& outputs = engine_->OutputBindings();
    const TensorBinding* features_binding = nullptr;
    for (const auto& b : outputs) {
        if (b.name == "last_hidden_state") {
            features_binding = &b;
            break;
        }
    }

    if (features_binding == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_InvalidBinding,
            "DinoV3Adapter: output binding 'last_hidden_state' not found",
        });
    }

    // 5. Validate output size.
    std::size_t actual_elements = features_binding->size_bytes / sizeof(float);
    if (actual_elements < output_floats) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_ModelConfigMismatch,
            "DinoV3Adapter: output size mismatch (expected " +
                       std::to_string(output_floats) + " floats, got " +
                       std::to_string(actual_elements) + ")",
        });
    }

    // 6. Skip CLS token (first token), return pure patch features
    return PatchFeatures{
        .device_ptr = static_cast<float*>(features_binding->device_ptr) + cfg_.embed_dim,
        .grid_h = grid_h,
        .grid_w = grid_w,
        .dim = cfg_.embed_dim,
    };
}

}  // namespace sai::inference
