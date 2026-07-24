// feature_bank.cpp — FeatureBank 可移植 CPU 实现（FAISS IndexFlatL2）
#include <sai/detection/feature_bank.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <source_location>
#include <span>
#include <string>
#include <string_view>

#include <faiss/IndexFlat.h>
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

namespace sai::detection {

#if defined(SAI_CUDA_ENABLED)
auto SelectGreedyCoresetCuda(
    const float*, std::size_t, std::size_t, std::size_t,
    std::vector<std::size_t>&) noexcept -> Result<void>;
#endif

namespace {

[[nodiscard]] auto CollectEmbeddingVectors(
    std::span<const sai::embedding::Embedding* const> embeddings,
    std::size_t dim,
    std::string_view operation) noexcept -> Result<std::vector<float>> {
    if (embeddings.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            std::string(operation) + ": no embeddings provided",
            std::source_location::current(),
        });
    }

    std::size_t total_values = 0;
    for (const auto* embedding : embeddings) {
        if (embedding == nullptr) continue;
        const auto& meta = embedding->Meta();
        if (meta.dim != dim) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_FeatureBankLoadFailed,
                std::string(operation) + ": embedding dim mismatch (expected "
                    + std::to_string(dim) + ", got " + std::to_string(meta.dim) + ")",
                std::source_location::current(),
            });
        }
        total_values += meta.count * dim;
    }
    if (total_values == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            std::string(operation) + ": no patch vectors extracted",
            std::source_location::current(),
        });
    }

    std::vector<float> vectors;
    vectors.reserve(total_values);
    for (const auto* embedding : embeddings) {
        if (embedding == nullptr || embedding->Meta().count == 0) continue;
        const auto value_count = embedding->Meta().count * dim;
        vectors.insert(vectors.end(), embedding->Data(),
                       embedding->Data() + value_count);
    }
    return vectors;
}

[[nodiscard]] auto SelectGreedyCoreset(
    const float* vectors, std::size_t count, std::size_t dim,
    std::size_t max_samples) noexcept -> Result<std::vector<std::size_t>> {
    if (vectors == nullptr || count == 0 || dim == 0 || max_samples == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "greedy coreset requires non-empty vectors, dimension and target",
            std::source_location::current(),
        });
    }
    if (count == 1 || max_samples == 1) return std::vector<std::size_t>{0};
#if defined(SAI_CUDA_ENABLED)
    std::vector<std::size_t> indices;
    auto result = SelectGreedyCoresetCuda(vectors, count, dim, max_samples, indices);
    if (!result) return tl::make_unexpected(result.error());
    return indices;
#else
    const auto target = std::min(count, max_samples);
    std::vector<std::size_t> indices;
    std::vector<float> min_distances(count, std::numeric_limits<float>::infinity());
    indices.reserve(target);
    indices.push_back(0);
    while (true) {
        const auto* selected = vectors + indices.back() * dim;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(count); ++n) {
            const auto i = static_cast<std::size_t>(n);
            const auto* candidate = vectors + i * dim;
            float distance = 0.0F;
            for (std::size_t d = 0; d < dim; ++d) {
                const auto delta = candidate[d] - selected[d];
                distance += delta * delta;
            }
            min_distances[i] = std::min(min_distances[i], distance);
        }
        if (indices.size() >= target) break;
        const auto best = std::max_element(min_distances.begin(), min_distances.end());
        if (*best <= 0.0F) break;
        indices.push_back(static_cast<std::size_t>(best - min_distances.begin()));
    }
    return indices;
#endif
}

}  // namespace

auto FeatureBank::LoadFromFile(const std::filesystem::path& path,
                               std::size_t dim) noexcept -> Result<FeatureBank> {
    // 读取文件
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "cannot open feature bank file: " + path.string(),
            std::source_location::current(),
        });
    }

    auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    auto element_size = sizeof(float);
    if (file_size % (dim * element_size) != 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "file size " + std::to_string(file_size)
                + " is not a multiple of dim " + std::to_string(dim),
            std::source_location::current(),
        });
    }

    auto num_samples = file_size / (dim * element_size);
    std::vector<float> buffer(num_samples * dim);
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size() * element_size));
    if (file.fail()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "failed to read feature bank file: " + path.string(),
            std::source_location::current(),
        });
    }

    // 创建 FAISS IndexFlatL2
    auto index = std::make_unique<faiss::IndexFlatL2>(static_cast<faiss::idx_t>(dim));
    index->add(static_cast<faiss::idx_t>(num_samples), buffer.data());

    FeatureBank bank;
    bank.index_ = std::move(index);
    bank.dim_ = dim;
    bank.num_samples_ = num_samples;
    return bank;
}

auto FeatureBank::Search(const float* query, std::size_t query_count,
                         std::size_t k) const noexcept -> std::vector<float> {
    if (query_count == 0 || k == 0 || !index_) {
        return {};
    }

    auto nq = static_cast<faiss::idx_t>(query_count);
    auto nk = static_cast<faiss::idx_t>(k);

    std::vector<float> distances(static_cast<std::size_t>(nq * nk));
    thread_local std::vector<faiss::idx_t> labels;
    labels.resize(static_cast<std::size_t>(nq * nk));

    // Dispatch to GPU FAISS index when CUDA+FAISS-GPU is available and
    // FeatureBank has been migrated via ToGpu(). Falls back to CPU otherwise.
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    auto* idx = (on_gpu_ && gpu_index_) ? gpu_index_.get() : index_.get();
#else
    auto* idx = index_.get();
#endif

    idx->search(nq, query, nk, distances.data(), labels.data());

    return distances;
}

auto FeatureBank::ExtractAllVectors() const noexcept -> std::vector<float> {
    if (num_samples_ == 0 || dim_ == 0 || !index_) {
        return {};
    }
    std::vector<float> result(num_samples_ * dim_);
    index_->reconstruct_n(0, static_cast<faiss::idx_t>(num_samples_), result.data());
    return result;
}

auto FeatureBank::Rebuild(const float* vectors, std::size_t count,
                          std::size_t dim) noexcept -> void {
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    gpu_index_.reset();
    gpu_resources_.reset();
    on_gpu_ = false;
#endif
    auto index = std::make_unique<faiss::IndexFlatL2>(static_cast<faiss::idx_t>(dim));
    index->add(static_cast<faiss::idx_t>(count), vectors);
    index_ = std::move(index);
    dim_ = dim;
    num_samples_ = count;
}

auto FeatureBank::SaveToFile(const std::filesystem::path& path) const noexcept -> Result<void> {
    if (!index_ || num_samples_ == 0 || dim_ == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "FeatureBank is empty, nothing to save",
            std::source_location::current(),
        });
    }

    auto vectors = ExtractAllVectors();
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "Cannot open file for writing: " + path.string(),
            std::source_location::current(),
        });
    }

    auto byte_count = vectors.size() * sizeof(float);
    file.write(reinterpret_cast<const char*>(vectors.data()),
               static_cast<std::streamsize>(byte_count));
    if (file.fail()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "Failed to write feature bank file: " + path.string(),
            std::source_location::current(),
        });
    }

    return {};
}

auto FeatureBank::BuildFromEmbeddings(
    std::span<const sai::embedding::Embedding* const> embeddings,
    std::size_t dim,
    std::size_t max_samples) noexcept -> Result<FeatureBank> {
    auto vectors_result = CollectEmbeddingVectors(
        embeddings, dim, "BuildFromEmbeddings");
    if (!vectors_result) return tl::make_unexpected(vectors_result.error());
    auto all_vectors = std::move(*vectors_result);
    const auto total_patches = all_vectors.size() / dim;

    // Uniform subsampling if total exceeds max_samples
    std::size_t num_samples = total_patches;
    const float* sampled_data = all_vectors.data();

    if (total_patches > max_samples) {
        // Stride-based subsampling: take every Nth vector
        num_samples = max_samples;
        auto stride = static_cast<std::size_t>(
            static_cast<double>(total_patches) / static_cast<double>(max_samples));
        if (stride < 1) stride = 1;

        std::vector<float> subsampled;
        subsampled.reserve(num_samples * dim);
        for (std::size_t i = 0; i < total_patches && subsampled.size() / dim < max_samples; i += stride) {
            subsampled.insert(subsampled.end(),
                              all_vectors.data() + i * dim,
                              all_vectors.data() + (i + 1) * dim);
        }
        // Adjust to exact max_samples
        while (subsampled.size() / dim < max_samples && subsampled.size() / dim < total_patches) {
            auto idx = subsampled.size() / dim;
            subsampled.insert(subsampled.end(),
                              all_vectors.data() + idx * dim,
                              all_vectors.data() + (idx + 1) * dim);
        }
        all_vectors = std::move(subsampled);
        sampled_data = all_vectors.data();
        num_samples = all_vectors.size() / dim;
    }

    FeatureBank bank;
    bank.Rebuild(sampled_data, num_samples, dim);
    return bank;
}

auto FeatureBank::BuildGreedyFromVectors(
    std::span<const float> vectors,
    std::size_t dim,
    std::size_t max_samples) noexcept -> Result<FeatureBank> {
    if (dim == 0 || vectors.empty() || vectors.size() % dim != 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildGreedyFromVectors: invalid vector matrix",
            std::source_location::current(),
        });
    }
    if (!std::all_of(vectors.begin(), vectors.end(), [](float value) {
            return std::isfinite(value);
        })) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildGreedyFromVectors: vectors must be finite",
            std::source_location::current(),
        });
    }
    const auto total_patches = vectors.size() / dim;

    auto indices_result = SelectGreedyCoreset(
        vectors.data(), total_patches, dim, max_samples);
    if (!indices_result) return tl::make_unexpected(indices_result.error());
    auto indices = std::move(*indices_result);
    const auto num_samples = indices.size();

    // Build output vector from selected indices.
    std::vector<float> coreset_data;
    coreset_data.reserve(num_samples * dim);
    for (auto idx : indices) {
        coreset_data.insert(coreset_data.end(),
                            vectors.data() + idx * dim,
                            vectors.data() + (idx + 1) * dim);
    }

    FeatureBank bank;
    bank.Rebuild(coreset_data.data(), num_samples, dim);
    return bank;
}

auto FeatureBank::BuildFromVectors(const float* vectors,
                                   std::size_t count,
                                   std::size_t dim) noexcept -> FeatureBank {
    FeatureBank bank;
    bank.Rebuild(vectors, count, dim);
    return bank;
}

#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
auto FeatureBank::ToGpu(int device) noexcept -> Result<void> {
    if (!index_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "FeatureBank::ToGpu: no index loaded",
            std::source_location::current(),
        });
    }

    if (on_gpu_) return {};

    try {
        gpu_resources_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        gpu_resources_->setTempMemory(64 * 1024 * 1024);
        gpu_index_ = std::unique_ptr<faiss::Index>(
            faiss::gpu::index_cpu_to_gpu(gpu_resources_.get(), device, index_.get()));
        on_gpu_ = true;
        return {};
    } catch (const std::exception& e) {
        gpu_index_.reset();
        gpu_resources_.reset();
        on_gpu_ = false;
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
#endif

FeatureBank::FeatureBank(FeatureBank&&) noexcept = default;
auto FeatureBank::operator=(FeatureBank&&) noexcept -> FeatureBank& = default;

FeatureBank::~FeatureBank() = default;

FeatureBank::FeatureBank() noexcept = default;

}  // namespace sai::detection
