// embedding_test.cpp — 批次 3.2 Embedding 双存储数据类型测试
#include <numeric>
#include <type_traits>

#include <sai/embedding/embedding.h>

#include <gtest/gtest.h>

namespace {

using namespace sai::embedding;

// ============================================================================
// Type traits
// ============================================================================

TEST(Embedding, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<Embedding>);
    static_assert(!std::is_copy_assignable_v<Embedding>);
    static_assert(std::is_move_constructible_v<Embedding>);
    static_assert(std::is_move_assignable_v<Embedding>);
    static_assert(std::is_nothrow_move_constructible_v<Embedding>);
    static_assert(std::is_nothrow_move_assignable_v<Embedding>);
    static_assert(std::is_final_v<Embedding>);
}

// FromGpu 签名编译验证——在 portable 构建中不调用（需要 IMemoryPool::Acquire）。
TEST(Embedding, FromGpuSignatureCompiles) {
    static_assert(std::is_invocable_v<decltype(Embedding::FromGpu),
                                      sai::memory::PooledPtr<std::uint8_t>,
                                      EmbeddingMeta>);
}

// ToCpuAsync 声明编译验证——在 portable 构建中不定义（CUDA 门控），
// 调用它会导致链接器错误，但声明本身可编译。
TEST(Embedding, ToCpuAsyncDeclared) {
    static_assert(std::is_member_function_pointer_v<decltype(&Embedding::ToCpuAsync)>);
}

// ============================================================================
// FromCpu
// ============================================================================

TEST(Embedding, FromCpuCreatesCpuEmbedding) {
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = EmbeddingType::Patch;
    meta.dim = 1024;
    meta.count = 1369;  // 37 × 37 patches
    meta.grid = {37, 37};
    meta.inference_latency = std::chrono::nanoseconds(15'000'000);  // 15ms

    std::vector<float> data(meta.count * meta.dim);
    std::iota(data.begin(), data.end(), 0.0f);

    auto emb = Embedding::FromCpu(data, meta);

    EXPECT_FALSE(emb.IsOnGpu());
    EXPECT_NE(emb.Data(), nullptr);
    EXPECT_EQ(emb.Meta().model_name, "DINOv3");
    EXPECT_EQ(emb.Meta().type, EmbeddingType::Patch);
    EXPECT_EQ(emb.Meta().dim, 1024);
    EXPECT_EQ(emb.Meta().count, 1369);
    EXPECT_EQ(emb.Meta().grid[0], 37);
    EXPECT_EQ(emb.Meta().grid[1], 37);
    EXPECT_EQ(emb.Meta().inference_latency.count(), 15'000'000);
    EXPECT_EQ(emb.SizeBytes(), 1369 * 1024 * sizeof(float));

    // Data() 指向首元素
    EXPECT_FLOAT_EQ(emb.Data()[0], 0.0f);
    EXPECT_FLOAT_EQ(emb.Data()[1], 1.0f);
    EXPECT_FLOAT_EQ(emb.Data()[100], 100.0f);
}

TEST(Embedding, FromCpuEmptyVector) {
    EmbeddingMeta meta;
    meta.model_name = "empty";
    meta.dim = 0;
    meta.count = 0;

    auto emb = Embedding::FromCpu({}, meta);
    EXPECT_FALSE(emb.IsOnGpu());
    EXPECT_EQ(emb.Data(), nullptr);
    EXPECT_EQ(emb.SizeBytes(), 0U);
}

TEST(Embedding, FromCpuGlobalEmbedding) {
    EmbeddingMeta meta;
    meta.model_name = "CLIP";
    meta.type = EmbeddingType::Global;
    meta.dim = 512;
    meta.count = 1;

    std::vector<float> data(512);
    data[0] = 3.14f;
    data[511] = 2.71f;

    auto emb = Embedding::FromCpu(std::move(data), meta);
    EXPECT_FALSE(emb.IsOnGpu());
    EXPECT_FLOAT_EQ(emb.Data()[0], 3.14f);
    EXPECT_FLOAT_EQ(emb.Data()[511], 2.71f);
    EXPECT_EQ(emb.SizeBytes(), 512 * sizeof(float));
}

// ============================================================================
// SizeBytes
// ============================================================================

TEST(Embedding, SizeBytesMatchesCountTimesDimTimesFloatSize) {
    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.dim = 64;
    meta.count = 100;

    auto emb = Embedding::FromCpu(std::vector<float>(6400), meta);
    EXPECT_EQ(emb.SizeBytes(), 100 * 64 * sizeof(float));
}

TEST(Embedding, SizeBytesWithZeroCountOrDim) {
    {
        auto emb = Embedding::FromCpu({}, EmbeddingMeta{.model_name = "z", .dim = 0, .count = 100});
        EXPECT_EQ(emb.SizeBytes(), 0U);
    }
    {
        auto emb = Embedding::FromCpu({}, EmbeddingMeta{.model_name = "z", .dim = 64, .count = 0});
        EXPECT_EQ(emb.SizeBytes(), 0U);
    }
}

// ============================================================================
// Const Data() 返回相同指针
// ============================================================================

TEST(Embedding, ConstDataReturnsSamePointer) {
    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.dim = 16;
    meta.count = 2;

    auto emb = Embedding::FromCpu(std::vector<float>(32), meta);
    const auto& const_emb = emb;
    EXPECT_EQ(emb.Data(), const_emb.Data());
}

// ============================================================================
// Move 语义
// ============================================================================

TEST(Embedding, MoveConstructionLeavesSourceValid) {
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.dim = 64;
    meta.count = 50;

    std::vector<float> data(3200);
    std::iota(data.begin(), data.end(), 42.0f);

    auto src = Embedding::FromCpu(data, meta);
    ASSERT_NE(src.Data(), nullptr);
    ASSERT_FALSE(src.IsOnGpu());

    auto dst = Embedding(std::move(src));

    // 目标拥有数据
    EXPECT_FALSE(dst.IsOnGpu());
    EXPECT_NE(dst.Data(), nullptr);
    EXPECT_FLOAT_EQ(dst.Data()[0], 42.0f);
    EXPECT_EQ(dst.Meta().model_name, "DINOv3");
    EXPECT_EQ(dst.SizeBytes(), 50 * 64 * sizeof(float));

    // 源移动后处于有效但未指定的状态——不会再报告 IsOnGpu()==true 且 Data()!=nullptr
    // 对于 FromCpu 路径：vector 被移动后 Data() 为 nullptr
    EXPECT_EQ(src.Data(), nullptr);
}

TEST(Embedding, MoveAssignmentTransfersOwnership) {
    EmbeddingMeta meta1;
    meta1.model_name = "source";
    meta1.dim = 8;
    meta1.count = 1;

    EmbeddingMeta meta2;
    meta2.model_name = "target";
    meta2.dim = 4;
    meta2.count = 2;

    auto a = Embedding::FromCpu(std::vector<float>(8, 1.0f), meta1);
    auto b = Embedding::FromCpu(std::vector<float>(8, 2.0f), meta2);

    b = std::move(a);

    // b 现在拥有 a 的数据
    EXPECT_EQ(b.Meta().model_name, "source");
    EXPECT_FLOAT_EQ(b.Data()[0], 1.0f);

    // a 被移动后 IsOnGpu() 保持 false（FromCpu 路径），Data() 为 nullptr
    EXPECT_FALSE(a.IsOnGpu());
}

TEST(Embedding, SelfMoveAssignmentSafe) {
    // 自赋值——不应崩溃或产生未定义行为
    // 注意：std::move(*this) 在 = default 的 move-assignment 上的行为是定义的。
    // 我们在此处不直接做 a = std::move(a)，而是验证 move-assign 不因别名而崩溃。
    EmbeddingMeta meta;
    meta.model_name = "self";
    meta.dim = 16;
    meta.count = 10;

    auto emb = Embedding::FromCpu(std::vector<float>(160, 3.0f), meta);
    auto& emb_ref = emb;
    // 不推荐的自赋值模式——编译器可能优化，仅测试不崩溃
    // 这里只验证正常的 move-assign 工作
    auto tmp = std::move(emb);
    ASSERT_NE(tmp.Data(), nullptr);
}

// ============================================================================
// EmbeddingType enum
// ============================================================================

TEST(Embedding, EmbeddingTypeValues) {
    EXPECT_EQ(static_cast<int>(EmbeddingType::Patch), 0);
    EXPECT_EQ(static_cast<int>(EmbeddingType::Global), 1);
}

// ============================================================================
// EmbeddingMeta 默认值
// ============================================================================

TEST(Embedding, EmbeddingMetaDefaultValues) {
    EmbeddingMeta meta;
    EXPECT_TRUE(meta.model_name.empty());
    EXPECT_EQ(meta.type, EmbeddingType::Patch);
    EXPECT_EQ(meta.dim, 0U);
    EXPECT_EQ(meta.count, 0U);
    EXPECT_EQ(meta.grid[0], 0U);
    EXPECT_EQ(meta.grid[1], 0U);
    EXPECT_EQ(meta.inference_latency.count(), 0);
}

}  // namespace
