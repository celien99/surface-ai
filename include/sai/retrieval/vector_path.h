// vector_path.h — 批次 4.2 FAISS 向量搜索路径
#pragma once

#include <cstddef>
#include <vector>

#include <sai/core/error.h>

namespace sai::detection {
class FeatureBank;
}  // namespace sai::detection

namespace sai::retrieval {

struct VectorResult {
    std::size_t index;
    float distance;
};

class VectorPath final {
public:
    enum class Mode { TopK, Range, Hybrid };

    struct Config {
        Mode mode = Mode::TopK;
        std::size_t k = 10;
        float range_threshold = 1.0F;
        std::vector<std::size_t> id_subset;
    };

    explicit VectorPath(const sai::detection::FeatureBank& bank) noexcept;

    [[nodiscard]] auto Search(const float* query, const Config& cfg) const noexcept
        -> Result<std::vector<VectorResult>>;

    [[nodiscard]] auto Dim() const noexcept -> std::size_t;

private:
    const sai::detection::FeatureBank& bank_;
};

}  // namespace sai::retrieval
