// feature_cache.h — 批次 3.2 FeatureCache LRU 特征缓存
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>

#include <sai/embedding/embedding.h>

namespace sai::embedding {

// FeatureCache——uint64_t 为键、Embedding 为值的 LRU 缓存，CPU 端。
//
// 设计决策：
// - LRU 淘汰：max_entries 满时淘汰最久未访问条目。Put 和 Get 均更新 LRU 顺序。
// - 线程安全：mutex 保护 LRU 结构；hits_/misses_ 为 atomic 无锁统计。
// - Embedding 为 move-only，Put 时通过移动语义传递所有权。
// - Get 返回 const Embedding*——指针在下次 Put（可能淘汰该条目）前有效。
//   调用方如需延长生命周期，自行拷贝 Embedding。
// - 仅仅 CPU 端：GPU 显存为稀缺资源，缓存使用廉价 CPU 内存。
class FeatureCache final {
public:
    // max_entries：最大缓存条目数。达到上限后，Put 新条目时淘汰 LRU 条目。
    explicit FeatureCache(std::size_t max_entries) noexcept : max_entries_(max_entries) {}

    // 查询缓存：命中返回指向 Embedding 的只读指针，未命中返回 nullptr。
    // 指针在下次 Put（可能淘汰该条目）之前有效。
    [[nodiscard]] auto Get(std::uint64_t key) noexcept -> const Embedding*;

    // 插入/更新缓存条目：移动 value 进缓存。
    // 如果 key 已存在，更新 LRU 位置；否则插入新条目。
    // 如果缓存已满（Size() == max_entries_），淘汰最久未访问条目后再插入。
    auto Put(std::uint64_t key, Embedding value) noexcept -> void;

    // 命中率 = hits / (hits + misses)。无访问时返回 0.0。
    [[nodiscard]] auto HitRate() const noexcept -> float;

    // 当前缓存条目数。
    [[nodiscard]] auto Size() const noexcept -> std::size_t;

    FeatureCache(const FeatureCache&) = delete;
    auto operator=(const FeatureCache&) -> FeatureCache& = delete;

private:
    using LruItem = std::pair<std::uint64_t, Embedding>;
    using LruIter = typename std::list<LruItem>::iterator;

    std::size_t max_entries_;
    std::list<LruItem> lru_;                                            // 前端 = 最近使用
    std::unordered_map<std::uint64_t, LruIter> index_;                  // key → list iterator
    std::atomic<std::size_t> hits_{0};
    std::atomic<std::size_t> misses_{0};
    mutable std::mutex mutex_;
};

}  // namespace sai::embedding
