#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sai::detection {

// Fixed-seed reservoir sampler. Memory is bounded by capacity * dim and the
// result depends only on input order, not on how input is split across Add().
class BoundedPatchSampler final {
public:
    BoundedPatchSampler(std::size_t dim, std::size_t capacity) noexcept;

    auto Add(const float* vectors, std::size_t count) -> void;

    [[nodiscard]] auto Size() const noexcept -> std::size_t;
    [[nodiscard]] auto Vectors() const noexcept -> const std::vector<float>&;

private:
    [[nodiscard]] auto NextRandom() noexcept -> std::uint64_t;

    std::size_t dim_;
    std::size_t capacity_;
    std::size_t seen_ = 0;
    std::uint64_t random_state_ = 0x6a09e667f3bcc909ULL;
    std::vector<float> vectors_;
};

}  // namespace sai::detection
