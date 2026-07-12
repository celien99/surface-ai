// feature_cache.cpp — 批次 3.2 FeatureCache LRU 特征缓存实现
#include <sai/embedding/feature_cache.h>

namespace sai::embedding {

auto FeatureCache::Get(std::uint64_t key) noexcept -> const Embedding* {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    // 命中 —— 将条目移到 list 前端（最近使用）
    hits_.fetch_add(1, std::memory_order_relaxed);
    lru_.splice(lru_.begin(), lru_, it->second);
    return &(it->second->second);
}

void FeatureCache::Put(std::uint64_t key, Embedding value) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(key);
    if (it != index_.end()) {
        // key 已存在 —— 更新值并移到前端
        it->second->second = std::move(value);
        lru_.splice(lru_.begin(), lru_, it->second);
        return;
    }

    // key 不存在 —— 确保有空间（淘汰 LRU 条目）
    if (lru_.size() >= max_entries_ && max_entries_ > 0) {
        auto& last = lru_.back();
        index_.erase(last.first);
        lru_.pop_back();
    } else if (max_entries_ == 0) {
        // max_entries == 0 时不缓存任何条目
        return;
    }

    // 插入新条目到前端
    lru_.emplace_front(key, std::move(value));
    index_[key] = lru_.begin();
}

auto FeatureCache::HitRate() const noexcept -> float {
    auto hits = hits_.load(std::memory_order_relaxed);
    auto misses = misses_.load(std::memory_order_relaxed);
    auto total = hits + misses;
    if (total == 0) {
        return 0.0f;
    }
    return static_cast<float>(hits) / static_cast<float>(total);
}

auto FeatureCache::Size() const noexcept -> std::size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return lru_.size();
}

}  // namespace sai::embedding
