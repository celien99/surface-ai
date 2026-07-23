#include <sai/inference/dino_v3_adapter.h>

#include <sai/image/gpu_image.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <exception>
#include <source_location>
#include <string>

namespace sai::inference {
namespace {

template <typename T>
__device__ auto Cast(float value) -> T;
template <>
__device__ auto Cast<float>(float value) -> float { return value; }
template <>
__device__ auto Cast<__half>(float value) -> __half { return __float2half(value); }

template <typename T>
__global__ void NormalizeRgb8(
    const std::uint8_t* input, std::size_t pixels, T* output) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float inv_std[3] = {
        1.0F / 0.229F, 1.0F / 0.224F, 1.0F / 0.225F};
    for (std::size_t c = 0; c < 3; ++c) {
        const auto value = static_cast<float>(input[i * 3 + c]) / 255.0F;
        output[c * pixels + i] = Cast<T>((value - mean[c]) * inv_std[c]);
    }
}

__global__ void HalfPatchesToFloat(
    const __half* tokens, std::size_t elements, std::size_t dim, float* patches) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) patches[i] = __half2float(tokens[dim + i]);
}

[[nodiscard]] auto CudaResult(
    cudaError_t error, const char* operation) noexcept -> Result<void> {
    if (error == cudaSuccess) return {};
    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Inference_EngineExecutionFailed,
        std::string(operation) + ": " + cudaGetErrorString(error),
        std::source_location::current(),
    });
}

[[nodiscard]] auto Allocate(
    std::size_t bytes, const char* name, void*& buffer) noexcept -> Result<void> {
    return CudaResult(cudaMalloc(&buffer, bytes), name);
}

[[nodiscard]] auto ValidateImage(
    const sai::image::GpuImage& image, std::size_t image_size) noexcept
    -> Result<void> {
    const auto& meta = image.Meta();
    if (meta.width == image_size && meta.height == image_size &&
        meta.channels == 3 && meta.pixel_format == sai::image::PixelFormat::RGB8 &&
        image.SizeBytes() == image_size * image_size * 3) {
        return {};
    }
    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Inference_ModelConfigMismatch,
        "DinoV3Adapter expects tightly packed RGB8 HWC at configured image size",
        std::source_location::current(),
    });
}

[[nodiscard]] auto Preprocess(
    const std::uint8_t* input, std::size_t pixels, TensorDataType dtype,
    void* output, cudaStream_t stream) noexcept -> Result<void> {
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned int>((pixels + threads - 1) / threads);
    if (dtype == TensorDataType::Float16) {
        NormalizeRgb8<<<blocks, threads, 0, stream>>>(
            input, pixels, static_cast<__half*>(output));
    } else {
        NormalizeRgb8<<<blocks, threads, 0, stream>>>(
            input, pixels, static_cast<float*>(output));
    }
    return CudaResult(cudaGetLastError(), "DINO preprocess kernel failed");
}

}  // namespace

auto DinoV3Adapter::Initialize() noexcept -> Result<void> {
    ReleaseBuffers();
    const auto grid = cfg_.image_size / cfg_.patch_size;
    auto result = Allocate(
        cfg_.image_size * cfg_.image_size * 3 * TensorDataTypeSize(input_dtype_),
        "DINO input allocation failed", input_buffer_);
    if (!result) return result;
    result = Allocate(
        (grid * grid + 1) * cfg_.embed_dim * TensorDataTypeSize(output_dtype_),
        "DINO output allocation failed", raw_output_buffer_);
    if (!result) {
        ReleaseBuffers();
        return result;
    }
    if (output_dtype_ == TensorDataType::Float16) {
        void* patches = nullptr;
        result = Allocate(grid * grid * cfg_.embed_dim * sizeof(float),
                          "DINO patch allocation failed", patches);
        if (!result) {
            ReleaseBuffers();
            return result;
        }
        patch_output_buffer_ = static_cast<float*>(patches);
    }
    result = engine_->SetTensorAddress("pixel_values", input_buffer_);
    if (!result) {
        ReleaseBuffers();
        return result;
    }
    result = engine_->SetTensorAddress("last_hidden_state", raw_output_buffer_);
    if (!result) ReleaseBuffers();
    return result;
}

auto DinoV3Adapter::ReleaseBuffers() noexcept -> void {
    cudaFree(patch_output_buffer_);
    cudaFree(raw_output_buffer_);
    cudaFree(input_buffer_);
    patch_output_buffer_ = nullptr;
    raw_output_buffer_ = nullptr;
    input_buffer_ = nullptr;
}

DinoV3Adapter::~DinoV3Adapter() { ReleaseBuffers(); }

DinoV3Adapter::DinoV3Adapter(DinoV3Adapter&& other) noexcept
    : engine_(other.engine_), cfg_(std::move(other.cfg_)),
      input_dtype_(other.input_dtype_), output_dtype_(other.output_dtype_),
      input_buffer_(other.input_buffer_),
      raw_output_buffer_(other.raw_output_buffer_),
      patch_output_buffer_(other.patch_output_buffer_) {
    other.input_buffer_ = nullptr;
    other.raw_output_buffer_ = nullptr;
    other.patch_output_buffer_ = nullptr;
}

auto DinoV3Adapter::operator=(DinoV3Adapter&& other) noexcept -> DinoV3Adapter& {
    if (this == &other) return *this;
    ReleaseBuffers();
    engine_ = other.engine_;
    cfg_ = std::move(other.cfg_);
    input_dtype_ = other.input_dtype_;
    output_dtype_ = other.output_dtype_;
    input_buffer_ = other.input_buffer_;
    raw_output_buffer_ = other.raw_output_buffer_;
    patch_output_buffer_ = other.patch_output_buffer_;
    other.input_buffer_ = nullptr;
    other.raw_output_buffer_ = nullptr;
    other.patch_output_buffer_ = nullptr;
    return *this;
}

auto DinoV3Adapter::InferImpl(
    const sai::image::GpuImage& image, void* stream, bool synchronize) noexcept
    -> Result<PatchFeatures> {
    auto result = ValidateImage(image, cfg_.image_size);
    if (!result) return tl::make_unexpected(result.error());
    if (input_buffer_ == nullptr || raw_output_buffer_ == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "DinoV3Adapter is not initialized",
            std::source_location::current(),
        });
    }

    auto cuda_stream = static_cast<cudaStream_t>(stream);
    result = Preprocess(image.Data(), cfg_.image_size * cfg_.image_size,
                        input_dtype_, input_buffer_, cuda_stream);
    if (!result) return tl::make_unexpected(result.error());
    result = engine_->InferAsync(stream);
    if (!result) return tl::make_unexpected(result.error());

    const auto grid = cfg_.image_size / cfg_.patch_size;
    float* patches = static_cast<float*>(raw_output_buffer_) + cfg_.embed_dim;
    if (output_dtype_ == TensorDataType::Float16) {
        constexpr int threads = 256;
        const auto elements = grid * grid * cfg_.embed_dim;
        const auto blocks = static_cast<unsigned int>((elements + threads - 1) / threads);
        HalfPatchesToFloat<<<blocks, threads, 0, cuda_stream>>>(
            static_cast<const __half*>(raw_output_buffer_), elements,
            cfg_.embed_dim, patch_output_buffer_);
        result = CudaResult(cudaGetLastError(), "DINO output conversion failed");
        if (!result) return tl::make_unexpected(result.error());
        patches = patch_output_buffer_;
    }
    if (synchronize) {
        result = CudaResult(cudaStreamSynchronize(cuda_stream),
                            "DINO output synchronization failed");
        if (!result) return tl::make_unexpected(result.error());
    }
    return PatchFeatures{patches, grid, grid, cfg_.embed_dim};
}

auto DinoV3Adapter::Infer(const sai::image::GpuImage& image) noexcept
    -> Result<PatchFeatures> {
    return InferImpl(image, nullptr, true);
}

auto DinoV3Adapter::InferAsync(
    const sai::image::GpuImage& image, void* stream) noexcept
    -> Result<PatchFeatures> {
    return InferImpl(image, stream, false);
}

}  // namespace sai::inference
