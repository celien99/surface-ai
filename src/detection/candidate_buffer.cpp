// candidate_buffer.cpp — 有界候选缓冲，线程安全
#include <sai/detection/coreset_evolution.h>

#include <mutex>
#include <vector>

namespace sai::detection {

auto CandidateBuffer::Append(EvolutionCandidate candidate) -> bool {
    std::lock_guard lock(mutex_);

    auto patch_count = candidate.grid_h * candidate.grid_w;
    if (candidates_.size() >= cfg_.max_frames
        || total_patches_ + patch_count > cfg_.max_patches) {
        return false;  // 缓冲区满
    }

    total_patches_ += patch_count;
    candidates_.push_back(std::move(candidate));
    return true;
}

auto CandidateBuffer::IsTriggered() const -> bool {
    std::lock_guard lock(mutex_);
    return candidates_.size() >= cfg_.trigger_frames
        || total_patches_ >= cfg_.trigger_patches;
}

auto CandidateBuffer::DrainAll() -> std::vector<EvolutionCandidate> {
    std::lock_guard lock(mutex_);
    total_patches_ = 0;
    return std::move(candidates_);
}

auto CandidateBuffer::FrameCount() const -> std::size_t {
    std::lock_guard lock(mutex_);
    return candidates_.size();
}

auto CandidateBuffer::PatchCount() const -> std::size_t {
    std::lock_guard lock(mutex_);
    return total_patches_;
}

}  // namespace sai::detection
