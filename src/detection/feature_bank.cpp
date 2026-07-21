// feature_bank.cpp — FeatureBank 可移植 CPU 实现（FAISS IndexFlatL2）
#include <sai/detection/feature_bank.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <source_location>
#include <span>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#endif

namespace sai::detection {

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

    // Configure nprobe for IVFFlat indices — dynamic_cast is safe here
    // because this is on the hot path and the cast is a simple vtable check.
    if (auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(idx)) {
        ivf->nprobe = static_cast<std::size_t>(nprobe_);
    }

    idx->search(nq, query, nk, distances.data(), labels.data());

    return distances;
}

auto FeatureBank::ExtractAllVectors() const noexcept -> std::vector<float> {
    if (num_samples_ == 0 || dim_ == 0 || !index_) {
        return {};
    }
    std::vector<float> result(num_samples_ * dim_);
    for (std::size_t i = 0; i < num_samples_; ++i) {
        index_->reconstruct(static_cast<faiss::idx_t>(i), result.data() + i * dim_);
    }
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
    if (embeddings.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildFromEmbeddings: no embeddings provided",
            std::source_location::current(),
        });
    }

    // Collect all patch vectors from all embeddings
    std::vector<float> all_vectors;
    std::size_t total_patches = 0;

    for (const auto* emb : embeddings) {
        if (emb == nullptr) continue;
        const auto& meta = emb->Meta();
        if (meta.dim != dim) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_FeatureBankLoadFailed,
                "BuildFromEmbeddings: embedding dim mismatch (expected "
                    + std::to_string(dim) + ", got " + std::to_string(meta.dim) + ")",
                std::source_location::current(),
            });
        }
        auto count = meta.count;
        if (count == 0) continue;
        const float* data = emb->Data();
        all_vectors.insert(all_vectors.end(), data, data + count * dim);
        total_patches += count;
    }

    if (total_patches == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildFromEmbeddings: no patch vectors extracted from embeddings",
            std::source_location::current(),
        });
    }

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

auto FeatureBank::BuildWithGreedyCoreset(
    std::span<const sai::embedding::Embedding* const> embeddings,
    std::size_t dim,
    std::size_t max_samples) noexcept -> Result<FeatureBank> {
    if (embeddings.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildWithGreedyCoreset: no embeddings provided",
            std::source_location::current(),
        });
    }

    // Collect all patch vectors and validate dimensions.
    std::vector<float> all_vectors;
    std::size_t total_patches = 0;

    for (const auto* emb : embeddings) {
        if (emb == nullptr) continue;
        const auto& meta = emb->Meta();
        if (meta.dim != dim) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_FeatureBankLoadFailed,
                "BuildWithGreedyCoreset: embedding dim mismatch (expected "
                    + std::to_string(dim) + ", got " + std::to_string(meta.dim) + ")",
                std::source_location::current(),
            });
        }
        auto count = meta.count;
        if (count == 0) continue;
        const float* data = emb->Data();
        all_vectors.insert(all_vectors.end(), data, data + count * dim);
        total_patches += count;
    }

    if (total_patches == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildWithGreedyCoreset: no patch vectors extracted",
            std::source_location::current(),
        });
    }

    auto num_samples = std::min(max_samples, total_patches);

    // ── Greedy furthest-point sampling ──
    // min_dist[i] = squared L2 distance from patch i to nearest coreset point.
    std::vector<float> min_dist(total_patches, std::numeric_limits<float>::max());
    std::vector<std::size_t> coreset_indices;
    coreset_indices.reserve(num_samples);

#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    auto selected_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
    auto selected_cpu = std::make_unique<faiss::IndexFlatL2>(
        static_cast<faiss::idx_t>(dim));
#endif
    std::unique_ptr<faiss::Index> selected_index;
    std::vector<faiss::idx_t> labels(total_patches);
    try {
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
        selected_resources->setTempMemory(512 * 1024 * 1024);
        selected_cpu->add(1, all_vectors.data());
        selected_index = std::unique_ptr<faiss::Index>(faiss::gpu::index_cpu_to_gpu(
            selected_resources.get(), 0, selected_cpu.get()));
#else
        selected_index = std::make_unique<faiss::IndexFlatL2>(
            static_cast<faiss::idx_t>(dim));
        selected_index->add(1, all_vectors.data());
#endif

        coreset_indices.push_back(0);
        selected_index->search(static_cast<faiss::idx_t>(total_patches),
                               all_vectors.data(), 1,
                               min_dist.data(), labels.data());

        // Iteratively select furthest point from current coreset.
        for (std::size_t k = 1; k < num_samples; ++k) {
            std::size_t best_idx = 0;
            float best_dist = -1.0F;
            for (std::size_t i = 0; i < total_patches; ++i) {
                if (min_dist[i] > best_dist) {
                    best_dist = min_dist[i];
                    best_idx = i;
                }
            }

            coreset_indices.push_back(best_idx);
            selected_index->add(1, all_vectors.data() + best_idx * dim);
            selected_index->search(static_cast<faiss::idx_t>(total_patches),
                                   all_vectors.data(), 1,
                                   min_dist.data(), labels.data());
        }
    } catch (const std::exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            std::string("BuildWithGreedyCoreset FAISS selection failed: ") + e.what(),
            std::source_location::current(),
        });
    }

    // Build output vector from selected indices.
    std::vector<float> coreset_data;
    coreset_data.reserve(num_samples * dim);
    for (auto idx : coreset_indices) {
        coreset_data.insert(coreset_data.end(),
                            all_vectors.data() + idx * dim,
                            all_vectors.data() + (idx + 1) * dim);
    }

    // Compute coverage statistics.
    float min_coverage = std::numeric_limits<float>::max();
    float max_coverage = 0.0F;
    double mean_coverage = 0.0;
    for (std::size_t i = 0; i < total_patches; ++i) {
        float d = std::sqrt(min_dist[i]);
        if (d < min_coverage) min_coverage = d;
        if (d > max_coverage) max_coverage = d;
        mean_coverage += static_cast<double>(d);
    }
    mean_coverage /= static_cast<double>(total_patches);

    // Log coverage stats (informational, not error).
    (void)min_coverage;
    (void)max_coverage;
    (void)mean_coverage;

    FeatureBank bank;
    bank.Rebuild(coreset_data.data(), num_samples, dim);
    return bank;
}

FeatureBank::FeatureBank(FeatureBank&&) noexcept = default;
auto FeatureBank::operator=(FeatureBank&&) noexcept -> FeatureBank& = default;

FeatureBank::~FeatureBank() = default;

FeatureBank::FeatureBank() noexcept = default;

// ────────────────────────────────────────────────────────────────────
// IVFFlat support: inverted index with K-means clustering
// ────────────────────────────────────────────────────────────────────

auto FeatureBank::BuildWithIVF(
    std::span<const sai::embedding::Embedding* const> embeddings,
    std::size_t dim,
    std::size_t max_samples,
    std::size_t nlist) noexcept -> Result<FeatureBank> {
    // First, collect all vectors using the same logic as BuildFromEmbeddings
    if (embeddings.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildWithIVF: no embeddings provided",
            std::source_location::current(),
        });
    }

    std::vector<float> all_vectors;
    std::size_t total_patches = 0;

    for (const auto* emb : embeddings) {
        if (emb == nullptr) continue;
        const auto& meta = emb->Meta();
        if (meta.dim != dim) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_FeatureBankLoadFailed,
                "BuildWithIVF: embedding dim mismatch (expected "
                    + std::to_string(dim) + ", got " + std::to_string(meta.dim) + ")",
                std::source_location::current(),
            });
        }
        auto count = meta.count;
        if (count == 0) continue;
        const float* data = emb->Data();
        all_vectors.insert(all_vectors.end(), data, data + count * dim);
        total_patches += count;
    }

    if (total_patches == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildWithIVF: no patch vectors extracted from embeddings",
            std::source_location::current(),
        });
    }

    // Uniform subsampling if total exceeds max_samples
    std::size_t num_samples = total_patches;
    const float* sampled_data = all_vectors.data();
    std::vector<float> subsampled;

    if (total_patches > max_samples) {
        num_samples = max_samples;
        auto stride = static_cast<std::size_t>(
            static_cast<double>(total_patches) / static_cast<double>(max_samples));
        if (stride < 1) stride = 1;

        subsampled.reserve(num_samples * dim);
        for (std::size_t i = 0;
             i < total_patches && subsampled.size() / dim < max_samples;
             i += stride) {
            subsampled.insert(subsampled.end(),
                              all_vectors.data() + i * dim,
                              all_vectors.data() + (i + 1) * dim);
        }
        sampled_data = subsampled.data();
        num_samples = subsampled.size() / dim;
    }

    // Clamp nlist: FAISS requires nlist <= num_samples for training.
    // For very small banks, fall back to a flat index.
    auto effective_nlist = std::min(nlist, num_samples);
    if (effective_nlist < 2) {
        // Too few samples for clustering — fall back to flat index
        FeatureBank bank;
        bank.Rebuild(sampled_data, num_samples, dim);
        return bank;
    }

    // Create IVFFlat quantizer + index
    auto quantizer =
        std::make_unique<faiss::IndexFlatL2>(static_cast<faiss::idx_t>(dim));
    auto index = std::make_unique<faiss::IndexIVFFlat>(
        quantizer.get(),
        static_cast<faiss::idx_t>(dim),
        static_cast<faiss::idx_t>(effective_nlist),
        faiss::METRIC_L2);

    // IndexIVFFlat takes ownership of quantizer — release unique_ptr
    index->own_fields = true;
    quantizer.release();

    // Train K-means on all vectors
    index->train(static_cast<faiss::idx_t>(num_samples), sampled_data);

    // Add vectors to inverted lists
    index->add(static_cast<faiss::idx_t>(num_samples), sampled_data);

    FeatureBank bank;
    bank.index_ = std::move(index);
    bank.dim_ = dim;
    bank.num_samples_ = num_samples;
    bank.nprobe_ = 4;
    return bank;
}

auto FeatureBank::ConvertToIVF(std::size_t nlist) noexcept -> Result<void> {
    if (!index_ || num_samples_ == 0 || dim_ == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "ConvertToIVF: FeatureBank is empty",
            std::source_location::current(),
        });
    }

    // Extract all vectors for training
    auto vectors = ExtractAllVectors();
    auto n = num_samples_;
    auto d = dim_;

    auto effective_nlist = std::min(nlist, n);
    if (effective_nlist < 2) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "ConvertToIVF: too few samples (" + std::to_string(n)
                + ") for clustering",
            std::source_location::current(),
        });
    }

    // Build IVFFlat index
    auto quantizer =
        std::make_unique<faiss::IndexFlatL2>(static_cast<faiss::idx_t>(d));
    auto new_index = std::make_unique<faiss::IndexIVFFlat>(
        quantizer.get(),
        static_cast<faiss::idx_t>(d),
        static_cast<faiss::idx_t>(effective_nlist),
        faiss::METRIC_L2);
    new_index->own_fields = true;
    quantizer.release();

    new_index->train(static_cast<faiss::idx_t>(n), vectors.data());
    new_index->add(static_cast<faiss::idx_t>(n), vectors.data());

#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    gpu_index_.reset();
    gpu_resources_.reset();
    on_gpu_ = false;
#endif
    index_ = std::move(new_index);
    return {};
}

}  // namespace sai::detection
