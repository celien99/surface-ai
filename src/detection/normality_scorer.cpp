// normality_scorer.cpp — 基于 coreset 自查询分布的帧级正常度评分
#include <sai/detection/coreset_evolution.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sai::detection {
namespace {

class NormalityScorer {
public:
    // 复用 PatchCore::Detect 已有的 k-NN distances，零额外 FAISS 查询
    static auto Assess(const float* distances,
                       std::size_t query_count,
                       const NormalityProfile& profile,
                       float tail_ratio_max) noexcept -> NormalityAssessment {
        if (query_count == 0 || profile.num_samples == 0) {
            return {};
        }

        // 1. 集中度——用 nth_element 求中位数 (O(M))
        std::vector<float> dists_copy(distances, distances + query_count);
        auto mid = dists_copy.begin() + static_cast<std::ptrdiff_t>(query_count / 2);
        std::nth_element(dists_copy.begin(), mid, dists_copy.end());
        float median_dist = *mid;

        float concentration = (profile.p50 > 0.0F)
            ? median_dist / profile.p50
            : 1.0F;

        // 2. 尾部比例——统计超过 P95 的 patch 数
        std::size_t tail_count = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            if (distances[i] > profile.p95) ++tail_count;
        }
        float tail_ratio = static_cast<float>(tail_count)
                         / static_cast<float>(query_count);

        // 3. 综合评分
        float score = 1.0F;
        if (tail_ratio > 0.0F && tail_ratio_max > 0.0F) {
            score = 1.0F - std::min(1.0F, tail_ratio / tail_ratio_max);
        }

        return {score, concentration, tail_ratio};
    }
};

}  // namespace
}  // namespace sai::detection
