// novelty_filter.cpp — 冗余剔除：只有覆盖稀疏区的帧才纳入候选
#include <sai/detection/coreset_evolution.h>

#include <algorithm>
#include <cstddef>

namespace sai::detection {
namespace {

class NoveltyFilter {
public:
    static auto Check(const float* distances,
                      std::size_t query_count,
                      const NormalityProfile& profile,
                      float coverage_threshold) noexcept -> NoveltyResult {
        NoveltyResult result;
        if (query_count == 0 || profile.num_samples == 0) return result;

        std::size_t covered = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            if (distances[i] < profile.p50) ++covered;
        }

        result.coverage_ratio = static_cast<float>(covered)
                              / static_cast<float>(query_count);
        result.novel_patch_count = query_count - covered;
        result.is_novel = (result.coverage_ratio < coverage_threshold);
        return result;
    }
};

}  // namespace
}  // namespace sai::detection
