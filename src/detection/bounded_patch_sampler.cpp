#include <sai/detection/bounded_patch_sampler.h>

#include <algorithm>

namespace sai::detection {

BoundedPatchSampler::BoundedPatchSampler(std::size_t dim,
                                         std::size_t capacity) noexcept
    : dim_(dim), capacity_(capacity) {
    vectors_.reserve(dim * capacity);
}

auto BoundedPatchSampler::Add(const float* vectors, std::size_t count) -> void {
    for (std::size_t i = 0; i < count; ++i) {
        const auto* patch = vectors + i * dim_;
        ++seen_;
        if (Size() < capacity_) {
            vectors_.insert(vectors_.end(), patch, patch + dim_);
            continue;
        }

        const auto slot = static_cast<std::size_t>(NextRandom() % seen_);
        if (slot < capacity_) {
            std::copy_n(patch, dim_, vectors_.data() + slot * dim_);
        }
    }
}

auto BoundedPatchSampler::Size() const noexcept -> std::size_t {
    return dim_ == 0 ? 0 : vectors_.size() / dim_;
}

auto BoundedPatchSampler::Vectors() const noexcept -> const std::vector<float>& {
    return vectors_;
}

auto BoundedPatchSampler::NextRandom() noexcept -> std::uint64_t {
    random_state_ += 0x9e3779b97f4a7c15ULL;
    auto value = random_state_;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace sai::detection
