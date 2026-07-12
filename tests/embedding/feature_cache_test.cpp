// feature_cache_test.cpp — 批次 3.2 FeatureCache LRU 特征缓存测试
#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

#include <sai/embedding/feature_cache.h>

#include <gtest/gtest.h>

namespace {

using namespace sai::embedding;
using sai::embedding::EmbeddingType;

// Helper: 创建一个指定数据的简单 Embedding（count=1, Global）
auto MakeTestEmbedding(float first_val, std::size_t dim = 64) -> Embedding {
    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.type = EmbeddingType::Global;
    meta.dim = dim;
    meta.count = 1;
    std::vector<float> data(dim);
    data[0] = first_val;
    // 填充其余元素使其可辨识
    for (std::size_t i = 1; i < dim; ++i) {
        data[i] = static_cast<float>(i);
    }
    return Embedding::FromCpu(std::move(data), meta);
}

// ============================================================================
// Type traits
// ============================================================================

TEST(FeatureCache, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<FeatureCache>);
    static_assert(!std::is_copy_assignable_v<FeatureCache>);
    static_assert(!std::is_move_constructible_v<FeatureCache>);
    static_assert(!std::is_move_assignable_v<FeatureCache>);
    static_assert(std::is_final_v<FeatureCache>);
}

// ============================================================================
// Get on empty cache returns nullptr
// ============================================================================

TEST(FeatureCache, GetEmptyReturnsNullptr) {
    FeatureCache cache(10);
    EXPECT_EQ(cache.Get(42), nullptr);
    EXPECT_EQ(cache.Get(0), nullptr);
    EXPECT_EQ(cache.Size(), 0U);
}

// ============================================================================
// Put then Get returns non-null matching Embedding
// ============================================================================

TEST(FeatureCache, PutAndGetReturnsNonNull) {
    FeatureCache cache(10);
    auto emb = MakeTestEmbedding(3.14f, 8);

    cache.Put(100, std::move(emb));
    EXPECT_EQ(cache.Size(), 1U);

    const Embedding* retrieved = cache.Get(100);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_FLOAT_EQ(retrieved->Data()[0], 3.14f);
}

// ============================================================================
// Get after Put for same key returns matching data
// ============================================================================

TEST(FeatureCache, PutAndGetReturnsMatchingEmbedding) {
    FeatureCache cache(10);

    {
        auto emb = MakeTestEmbedding(1.0f, 16);
        cache.Put(1, std::move(emb));
    }
    {
        auto emb = MakeTestEmbedding(2.0f, 16);
        cache.Put(2, std::move(emb));
    }

    const Embedding* e1 = cache.Get(1);
    ASSERT_NE(e1, nullptr);
    EXPECT_FLOAT_EQ(e1->Data()[0], 1.0f);

    const Embedding* e2 = cache.Get(2);
    ASSERT_NE(e2, nullptr);
    EXPECT_FLOAT_EQ(e2->Data()[0], 2.0f);
}

// ============================================================================
// LRU eviction when max_entries exceeded
// ============================================================================

TEST(FeatureCache, LRUEviction) {
    FeatureCache cache(2);  // 最多 2 个条目

    cache.Put(1, MakeTestEmbedding(1.0f));
    cache.Put(2, MakeTestEmbedding(2.0f));
    EXPECT_EQ(cache.Size(), 2U);

    // Put 第 3 个条目 → 淘汰 key=1（最久未访问）
    cache.Put(3, MakeTestEmbedding(3.0f));
    EXPECT_EQ(cache.Size(), 2U);

    // key=1 应被淘汰
    EXPECT_EQ(cache.Get(1), nullptr);

    // key=2 和 key=3 应存在
    EXPECT_NE(cache.Get(2), nullptr);
    EXPECT_NE(cache.Get(3), nullptr);
}

// ============================================================================
// LRU access order — getting an entry promotes it
// ============================================================================

TEST(FeatureCache, LRUAccessOrder) {
    FeatureCache cache(2);

    cache.Put(1, MakeTestEmbedding(1.0f));
    cache.Put(2, MakeTestEmbedding(2.0f));

    // 访问 key=1（更新 LRU 顺序）：1 变为最近使用，2 变为最久未访问
    EXPECT_NE(cache.Get(1), nullptr);

    // Put 第 3 个条目 → 淘汰 key=2（最久未访问）
    cache.Put(3, MakeTestEmbedding(3.0f));

    // key=1 应存在（被访问过），key=2 应被淘汰
    EXPECT_NE(cache.Get(1), nullptr);
    EXPECT_EQ(cache.Get(2), nullptr);
    EXPECT_NE(cache.Get(3), nullptr);
}

// ============================================================================
// HitRate with known access pattern
// ============================================================================

TEST(FeatureCache, HitRate) {
    FeatureCache cache(10);

    // 初始未访问
    EXPECT_FLOAT_EQ(cache.HitRate(), 0.0f);

    // 访问不存在的 key（miss）
    (void)cache.Get(1);    // 1 miss, 0 hit
    EXPECT_FLOAT_EQ(cache.HitRate(), 0.0f);

    // 插入并命中
    cache.Put(1, MakeTestEmbedding(1.0f));

    // 实际上，Put 后需要再次 Get 来命中
    cache.Put(2, MakeTestEmbedding(2.0f));
    (void)cache.Get(2);    // 1 hit, 1 miss (from Get(1) above)

    // hits=1, misses=1 → HitRate=0.5
    EXPECT_NEAR(cache.HitRate(), 0.5f, 1e-6f);
}

// ============================================================================
// HitRate with mix of hits and misses
// ============================================================================

TEST(FeatureCache, HitRateMix) {
    FeatureCache cache(10);

    // miss
    (void)cache.Get(1);
    // miss
    (void)cache.Get(2);
    // 插入 key=3
    cache.Put(3, MakeTestEmbedding(3.0f));
    // hit
    (void)cache.Get(3);
    // miss
    (void)cache.Get(4);
    // hit (again)
    (void)cache.Get(3);

    // hits=2, misses=3 → HitRate=0.4
    EXPECT_NEAR(cache.HitRate(), 0.4f, 1e-6f);
}

// ============================================================================
// HitRate with no accesses
// ============================================================================

TEST(FeatureCache, HitRateNoAccess) {
    FeatureCache cache(10);
    EXPECT_FLOAT_EQ(cache.HitRate(), 0.0f);
}

// ============================================================================
// Size starts at 0
// ============================================================================

TEST(FeatureCache, SizeStartsAtZero) {
    FeatureCache cache(5);
    EXPECT_EQ(cache.Size(), 0U);
}

// ============================================================================
// Size reflects number of entries
// ============================================================================

TEST(FeatureCache, SizeReflectsEntries) {
    FeatureCache cache(5);
    EXPECT_EQ(cache.Size(), 0U);

    cache.Put(1, MakeTestEmbedding(1.0f));
    EXPECT_EQ(cache.Size(), 1U);

    cache.Put(2, MakeTestEmbedding(2.0f));
    EXPECT_EQ(cache.Size(), 2U);

    cache.Put(3, MakeTestEmbedding(3.0f));
    EXPECT_EQ(cache.Size(), 3U);
}

// ============================================================================
// Size after LRU eviction decreases
// ============================================================================

TEST(FeatureCache, SizeAfterEviction) {
    FeatureCache cache(2);
    cache.Put(1, MakeTestEmbedding(1.0f));
    cache.Put(2, MakeTestEmbedding(2.0f));
    EXPECT_EQ(cache.Size(), 2U);

    // 触发淘汰
    cache.Put(3, MakeTestEmbedding(3.0f));
    EXPECT_EQ(cache.Size(), 2U);  // 仍为 2（淘汰 1，插入 3）
}

// ============================================================================
// Repeated Put on same key updates without growing
// ============================================================================

TEST(FeatureCache, RepeatedPutSameKey) {
    FeatureCache cache(10);

    auto emb1 = MakeTestEmbedding(1.0f, 8);
    cache.Put(5, std::move(emb1));
    EXPECT_EQ(cache.Size(), 1U);

    // 同一 key 再次 Put 应覆盖值
    auto emb2 = MakeTestEmbedding(99.0f, 8);
    cache.Put(5, std::move(emb2));
    EXPECT_EQ(cache.Size(), 1U);  // 条目数不变

    const Embedding* retrieved = cache.Get(5);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_FLOAT_EQ(retrieved->Data()[0], 99.0f);
}

// ============================================================================
// max_entries = 0 cache rejects all puts
// ============================================================================

TEST(FeatureCache, ZeroMaxEntriesStoresNothing) {
    FeatureCache cache(0);
    cache.Put(1, MakeTestEmbedding(1.0f));
    EXPECT_EQ(cache.Size(), 0U);
    EXPECT_EQ(cache.Get(1), nullptr);
}

// ============================================================================
// Concurrent access
// ============================================================================

TEST(FeatureCache, ConcurrentAccess) {
    FeatureCache cache(100);

    // 预先插入 key
    for (int i = 0; i < 50; ++i) {
        cache.Put(static_cast<std::uint64_t>(i), MakeTestEmbedding(static_cast<float>(i)));
    }

    std::atomic<std::size_t> total_get{0};
    std::atomic<std::size_t> total_hit{0};

    // 两个线程并发访问
    auto worker = [&](int start, int count) {
        for (int i = 0; i < count; ++i) {
            std::uint64_t key = static_cast<std::uint64_t>(start + i);
            if (cache.Get(key) != nullptr) {
                total_hit.fetch_add(1);
            }
            total_get.fetch_add(1);
        }
    };

    std::thread t1(worker, 0, 30);
    std::thread t2(worker, 20, 30);

    t1.join();
    t2.join();

    // 至少有一些命中
    EXPECT_GT(total_hit.load(), 0U);
}

}  // namespace
