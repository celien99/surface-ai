// feature_bank.cpp — FeatureBank 可移植 CPU 实现（FAISS IndexFlatL2）
#include <sai/detection/feature_bank.h>
#include <sai/detection/bounded_patch_sampler.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <source_location>
#include <span>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

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
    if (embeddings.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildFromEmbeddings: no embeddings provided",
            std::source_location::current(),
        });
    }

    BoundedPatchSampler sampler(dim, max_samples);

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
        sampler.Add(emb->Data(), count);
    }

    if (sampler.Size() == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "BuildFromEmbeddings: no patch vectors extracted from embeddings",
            std::source_location::current(),
        });
    }

    return BuildFromVectors(sampler.Vectors().data(), sampler.Size(), dim);
}

auto FeatureBank::BuildFromVectors(const float* vectors,
                                   std::size_t count,
                                   std::size_t dim) noexcept -> FeatureBank {
    FeatureBank bank;
    bank.Rebuild(vectors, count, dim);
    return bank;
}

FeatureBank::FeatureBank(FeatureBank&&) noexcept = default;
auto FeatureBank::operator=(FeatureBank&&) noexcept -> FeatureBank& = default;

FeatureBank::~FeatureBank() = default;

FeatureBank::FeatureBank() noexcept = default;

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
    nprobe_ = std::min<std::size_t>(4, effective_nlist);
    new_index->nprobe = nprobe_;

#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    gpu_index_.reset();
    gpu_resources_.reset();
    on_gpu_ = false;
#endif
    index_ = std::move(new_index);
    return {};
}

}  // namespace sai::detection
