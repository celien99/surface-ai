#include <sai/core/error.h>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <source_location>
#include <string>
#include <vector>

namespace sai::detection {
namespace {

__global__ void TransposeVectors(
    const float* row_major,
    std::size_t count,
    std::size_t dim,
    float* dimension_major) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count * dim) return;
    const auto candidate = i / dim;
    const auto component = i % dim;
    dimension_major[component * count + candidate] = row_major[i];
}

__global__ void UpdateMinimumSquaredDistances(
    const float* dimension_major,
    std::size_t count,
    std::size_t dim,
    std::size_t selected_index,
    float* minimum_distances) {
    extern __shared__ float selected[];
    for (std::size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        selected[d] = dimension_major[d * count + selected_index];
    }
    __syncthreads();

    const auto candidate_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (candidate_index >= count) return;
    float squared_distance = 0.0F;
    for (std::size_t d = 0; d < dim; ++d) {
        const auto delta = dimension_major[d * count + candidate_index] - selected[d];
        squared_distance += delta * delta;
    }
    minimum_distances[candidate_index] =
        fminf(minimum_distances[candidate_index], squared_distance);
}

[[nodiscard]] auto CudaError(
    cudaError_t error, const char* operation) noexcept -> Result<void> {
    if (error == cudaSuccess) return {};
    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Detection_FeatureBankLoadFailed,
        std::string(operation) + " failed: " + cudaGetErrorString(error),
        std::source_location::current(),
    });
}

}  // namespace

auto SelectGreedyCoresetCuda(
    const float* vectors,
    std::size_t count,
    std::size_t dim,
    std::size_t max_samples,
    std::vector<std::size_t>& indices) noexcept -> Result<void> {
    float* device_row_major = nullptr;
    float* device_dimension_major = nullptr;
    float* device_minimum_distances = nullptr;
    const auto vector_bytes = count * dim * sizeof(float);
    const auto distance_bytes = count * sizeof(float);

    auto allocation = CudaError(
        cudaMalloc(&device_row_major, vector_bytes), "greedy FPS upload allocation");
    if (!allocation) return tl::make_unexpected(allocation.error());
    allocation = CudaError(cudaMalloc(&device_dimension_major, vector_bytes),
                           "greedy FPS candidate allocation");
    if (!allocation) {
        cudaFree(device_row_major);
        return tl::make_unexpected(allocation.error());
    }
    allocation = CudaError(cudaMalloc(&device_minimum_distances, distance_bytes),
                           "greedy FPS distance allocation");
    if (!allocation) {
        cudaFree(device_dimension_major);
        cudaFree(device_row_major);
        return tl::make_unexpected(allocation.error());
    }

    auto cleanup = [&]() noexcept {
        cudaFree(device_minimum_distances);
        cudaFree(device_dimension_major);
        cudaFree(device_row_major);
    };
    auto copy = CudaError(cudaMemcpy(device_row_major, vectors, vector_bytes,
                                     cudaMemcpyHostToDevice),
                          "greedy FPS candidate upload");
    if (!copy) {
        cleanup();
        return tl::make_unexpected(copy.error());
    }

    constexpr int threads = 256;
    const auto vector_elements = count * dim;
    const auto transpose_blocks = static_cast<unsigned int>(
        (vector_elements + threads - 1) / threads);
    TransposeVectors<<<transpose_blocks, threads>>>(
        device_row_major, count, dim, device_dimension_major);
    auto transpose = CudaError(cudaGetLastError(),
                               "greedy FPS candidate transpose");
    if (!transpose) {
        cleanup();
        return tl::make_unexpected(transpose.error());
    }
    cudaFree(device_row_major);
    device_row_major = nullptr;

    try {
        auto minimum_begin = thrust::device_pointer_cast(device_minimum_distances);
        thrust::fill(thrust::device, minimum_begin, minimum_begin + count,
                     std::numeric_limits<float>::infinity());

        const auto target = std::min(count, max_samples);
        indices.clear();
        indices.reserve(target);
        indices.push_back(0);

        const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
        const auto shared_bytes = dim * sizeof(float);
        while (true) {
            UpdateMinimumSquaredDistances<<<blocks, threads, shared_bytes>>>(
                device_dimension_major, count, dim, indices.back(),
                device_minimum_distances);
            auto launch = CudaError(cudaGetLastError(),
                                    "greedy FPS distance kernel launch");
            if (!launch) {
                cleanup();
                return tl::make_unexpected(launch.error());
            }
            if (indices.size() >= target) break;

            // thrust::max_element returns the first maximum, preserving the
            // smallest-index tie break used by the reference implementation.
            auto best_it = thrust::max_element(
                thrust::device, minimum_begin, minimum_begin + count);
            const auto best_index = static_cast<std::size_t>(best_it - minimum_begin);
            float best_distance = 0.0F;
            auto best_copy = CudaError(
                cudaMemcpy(&best_distance, device_minimum_distances + best_index,
                           sizeof(float), cudaMemcpyDeviceToHost),
                "greedy FPS maximum distance download");
            if (!best_copy) {
                cleanup();
                return tl::make_unexpected(best_copy.error());
            }
            if (best_distance <= 0.0F) break;
            indices.push_back(best_index);
        }

        cleanup();
        return {};
    } catch (const std::exception& exception) {
        cleanup();
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            std::string("greedy FPS GPU selection failed: ") + exception.what(),
            std::source_location::current(),
        });
    }
}

}  // namespace sai::detection
