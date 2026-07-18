// vector_path.cpp — VectorPath FAISS 多模式搜索实现
#include <sai/retrieval/vector_path.h>

#include <algorithm>
#include <source_location>
#include <string>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <sai/detection/feature_bank.h>

namespace sai::retrieval {

VectorPath::VectorPath(const sai::detection::FeatureBank& bank) noexcept : bank_(bank) {}

auto VectorPath::Dim() const noexcept -> std::size_t { return bank_.Dim(); }

// 通用 TopK 搜索
static auto topk_search(faiss::Index* index, const float* query,
                        faiss::idx_t k) -> std::vector<VectorResult> {
    std::vector<float> distances(static_cast<std::size_t>(k));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(k));
    index->search(1, query, k, distances.data(), labels.data());
    std::vector<VectorResult> results;
    results.reserve(static_cast<std::size_t>(k));
    for (faiss::idx_t i = 0; i < k; ++i) {
        if (labels[i] < 0) break;
        results.push_back({static_cast<std::size_t>(labels[i]), distances[i]});
    }
    return results;
}

auto VectorPath::Search(const float* query, std::size_t dim, const Config& cfg) const noexcept
    -> Result<std::vector<VectorResult>> {
    if (dim != bank_.Dim()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_DimensionMismatch,
            "query dimension " + std::to_string(dim) +
                " does not match index dimension " + std::to_string(bank_.Dim()),
            std::source_location::current(),
        });
    }

    // Dispatch to GPU FAISS index when CUDA+FAISS-GPU is available and
    // FeatureBank has been migrated via ToGpu(). Falls back to CPU otherwise.
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
    auto* index = (bank_.on_gpu_ && bank_.gpu_index_) ? bank_.gpu_index_.get()
                                                       : bank_.index_.get();
#else
    auto* index = bank_.index_.get();
#endif
    if (!index || bank_.NumSamples() == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_EmptyIndex,
            "FeatureBank is empty",
            std::source_location::current(),
        });
    }

    // Configure nprobe for IVFFlat indices — must match FeatureBank::Search().
    // Without this, IVFFlat defaults to nprobe=1, dropping ~75% recall
    // when FeatureBank calls go through VectorPath (FactBuilder retrieval).
    if (auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index)) {
        ivf->nprobe = static_cast<size_t>(bank_.nprobe_);
    }

    switch (cfg.mode) {
    case Mode::TopK: {
        auto k = static_cast<faiss::idx_t>(std::min(cfg.k, bank_.NumSamples()));
        return topk_search(index, query, k);
    }
    case Mode::Range: {
        faiss::RangeSearchResult range_res(1);
        index->range_search(1, query, cfg.range_threshold, &range_res);
        std::vector<VectorResult> results;
        results.reserve(static_cast<std::size_t>(range_res.lims[1]));
        for (faiss::idx_t i = 0; i < range_res.lims[1]; ++i) {
            results.push_back(
                {static_cast<std::size_t>(range_res.labels[i]), range_res.distances[i]});
        }
        return results;
    }
    case Mode::Hybrid: {
        // 空 id_subset 退化为 TopK
        if (cfg.id_subset.empty()) {
            auto k = static_cast<faiss::idx_t>(std::min(cfg.k, bank_.NumSamples()));
            return topk_search(index, query, k);
        }
        auto k = static_cast<faiss::idx_t>(std::min(cfg.k, cfg.id_subset.size()));
        std::vector<faiss::idx_t> sel_ids;
        sel_ids.reserve(cfg.id_subset.size());
        for (auto id : cfg.id_subset) {
            sel_ids.push_back(static_cast<faiss::idx_t>(id));
        }
        faiss::IDSelectorArray selector(static_cast<faiss::idx_t>(sel_ids.size()),
                                         sel_ids.data());
        faiss::SearchParameters params;
        params.sel = &selector;
        std::vector<float> distances(static_cast<std::size_t>(k));
        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(k));
        index->search(1, query, k, distances.data(), labels.data(), &params);
        std::vector<VectorResult> results;
        results.reserve(static_cast<std::size_t>(k));
        for (faiss::idx_t i = 0; i < k; ++i) {
            if (labels[i] < 0) break;
            results.push_back({static_cast<std::size_t>(labels[i]), distances[i]});
        }
        return results;
    }
    }
    return std::vector<VectorResult>{};
}

}  // namespace sai::retrieval
