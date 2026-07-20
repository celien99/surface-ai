// embedder_test.cpp — 批次 3.2 IEmbedder + PatchEmbedder + GlobalEmbedder 测试
#include <sai/embedding/embedder.h>
#include <sai/image/surface_image.h>
#include <sai/inference/clip_adapter.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/inference/mock_engine.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace sai::embedding;
using sai::ErrorCode;
using sai::inference::ClipAdapter;
using sai::inference::ClipConfig;
using sai::inference::DinoV3Adapter;
using sai::inference::DinoV3Config;
using sai::inference::MockEngine;
using sai::inference::TensorBinding;
using sai::image::ImageMeta;
using sai::image::PixelFormat;
using sai::image::SurfaceImage;

// 创建一个简单的 CPU SurfaceImage（8x8 RGB8），用于测试 GPU guard。
auto MakeSurfaceImage() -> SurfaceImage {
    ImageMeta meta{.width = 8,
                   .height = 8,
                   .channels = 3,
                   .pixel_format = PixelFormat::RGB8};
    return SurfaceImage::FromOwnedBuffer(
        std::vector<std::uint8_t>(8 * 8 * 3, 128), meta);
}

// 构建 DinoV3Adapter——MockEngine 必须比返回的 adapter 长寿（adapter 持有其指针）。
// 调用方在同一作用域内保持 engine 存活即可。
auto BuildDinoV3Adapter(MockEngine& engine) -> DinoV3Adapter {
    auto outputs = std::vector<TensorBinding>{
        {"last_hidden_state", {1, 37, 37, 1024}, 0, nullptr},
    };
    auto load_ok = engine.Load("dino_v3.engine", {}, outputs);
    EXPECT_TRUE(load_ok.has_value());

    DinoV3Config cfg{.engine_path = "dino_v3.engine",
                     .image_size = 518,
                     .patch_size = 14,
                     .embed_dim = 1024};
    auto adapter = DinoV3Adapter::Create(engine, cfg);
    EXPECT_TRUE(adapter.has_value());
    return std::move(*adapter);
}

// 构建 ClipAdapter——MockEngine 必须比返回的 adapter 长寿。
auto BuildClipAdapter(MockEngine& engine) -> ClipAdapter {
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 512}, 0, nullptr},
    };
    auto load_ok = engine.Load("clip.engine", {}, outputs);
    EXPECT_TRUE(load_ok.has_value());

    ClipConfig cfg{.engine_path = "clip.engine",
                   .image_size = 224,
                   .embed_dim = 512};
    auto adapter = ClipAdapter::Create(engine, cfg);
    EXPECT_TRUE(adapter.has_value());
    return std::move(*adapter);
}

// ============================================================================
// Type traits
// ============================================================================

TEST(PatchEmbedder, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<PatchEmbedder>);
    static_assert(!std::is_copy_assignable_v<PatchEmbedder>);
    static_assert(std::is_move_constructible_v<PatchEmbedder>);
    static_assert(std::is_move_assignable_v<PatchEmbedder>);
    static_assert(std::is_final_v<PatchEmbedder>);
}

TEST(GlobalEmbedder, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<GlobalEmbedder>);
    static_assert(!std::is_copy_assignable_v<GlobalEmbedder>);
    static_assert(std::is_move_constructible_v<GlobalEmbedder>);
    static_assert(std::is_move_assignable_v<GlobalEmbedder>);
    static_assert(std::is_final_v<GlobalEmbedder>);
}

// IEmbedder 从 Object 派生，且 Object 不可复制/移动。
TEST(IEmbedder, IsPolymorphic) {
    static_assert(std::is_polymorphic_v<IEmbedder>);
    static_assert(std::has_virtual_destructor_v<IEmbedder>);
}

// ============================================================================
// PatchEmbedder::Create
// ============================================================================

TEST(PatchEmbedder, CreateSucceedsWithValidAdapter) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());
}

TEST(PatchEmbedder, ModelNameReturnsDinoV3) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());
    EXPECT_EQ(embedder->ModelName(), "DINOv2");
}

// ============================================================================
// GlobalEmbedder::Create
// ============================================================================

TEST(GlobalEmbedder, CreateSucceedsWithValidAdapter) {
    MockEngine engine;
    auto adapter = BuildClipAdapter(engine);
    auto embedder = GlobalEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());
}

TEST(GlobalEmbedder, ModelNameReturnsClip) {
    MockEngine engine;
    auto adapter = BuildClipAdapter(engine);
    auto embedder = GlobalEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());
    EXPECT_EQ(embedder->ModelName(), "CLIP");
}

// ============================================================================
// Extract GPU guard — SurfaceImage（CPU）应返回 Embedding_NotGpuImage
// ============================================================================

TEST(PatchEmbedder, ExtractSurfaceImageReturnsNotGpuImage) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    auto result = embedder->Extract(surface);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

TEST(GlobalEmbedder, ExtractSurfaceImageReturnsNotGpuImage) {
    MockEngine engine;
    auto adapter = BuildClipAdapter(engine);
    auto embedder = GlobalEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    auto result = embedder->Extract(surface);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

// ============================================================================
// ExtractBatch GPU guard
// ============================================================================

TEST(PatchEmbedder, ExtractBatchWithSurfaceImageReturnsNotGpuImage) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    const sai::image::Image* ptr = &surface;
    auto result = embedder->ExtractBatch(std::span<const sai::image::Image* const>(&ptr, 1));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

TEST(GlobalEmbedder, ExtractBatchWithSurfaceImageReturnsNotGpuImage) {
    MockEngine engine;
    auto adapter = BuildClipAdapter(engine);
    auto embedder = GlobalEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    const sai::image::Image* ptr = &surface;
    auto result = embedder->ExtractBatch(std::span<const sai::image::Image* const>(&ptr, 1));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

TEST(PatchEmbedder, ExtractBatchWithNullImagePointerReturnsNotGpuImage) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    const sai::image::Image* null_ptr = nullptr;
    auto result = embedder->ExtractBatch(std::span<const sai::image::Image* const>(&null_ptr, 1));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

TEST(PatchEmbedder, ExtractBatchEmptySpanSucceeds) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto result = embedder->ExtractBatch({});
    // 空 span 不经过 GPU guard 循环，直接进入 stub 路径——
    // 可移植构建返回 Inference_EngineExecutionFailed（无 CUDA 支持）
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Inference_EngineExecutionFailed);
}

// ============================================================================
// Move 语义：moved-from PatchEmbedder 不能 Extract
// ============================================================================

TEST(PatchEmbedder, MovedFromCannotExtract) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();

    // 移动前可 Extract（至少能执行 GPU guard）
    auto result_before = embedder->Extract(surface);
    ASSERT_FALSE(result_before.has_value());
    EXPECT_EQ(result_before.error().code, ErrorCode::Embedding_NotGpuImage);

    // 移动后，源不能 Extract
    auto moved_to = std::move(*embedder);
    auto result_after = embedder->Extract(surface);
    ASSERT_FALSE(result_after.has_value());
    EXPECT_EQ(result_after.error().code, ErrorCode::Inference_EngineExecutionFailed);
}

TEST(PatchEmbedder, MovedToCanExtract) {
    MockEngine engine;
    auto adapter = BuildDinoV3Adapter(engine);
    auto embedder = PatchEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    auto moved_to = std::move(*embedder);

    // 目标对象可 Extract（GPU guard 路径）
    auto result = moved_to.Extract(surface);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Embedding_NotGpuImage);
}

TEST(GlobalEmbedder, MovedFromCannotExtract) {
    MockEngine engine;
    auto adapter = BuildClipAdapter(engine);
    auto embedder = GlobalEmbedder::Create(std::move(adapter));
    ASSERT_TRUE(embedder.has_value());

    auto surface = MakeSurfaceImage();
    auto moved_to = std::move(*embedder);

    auto result_after = embedder->Extract(surface);
    ASSERT_FALSE(result_after.has_value());
    EXPECT_EQ(result_after.error().code, ErrorCode::Inference_EngineExecutionFailed);
}

// ============================================================================
// Move assignment
// ============================================================================

TEST(PatchEmbedder, MoveAssignmentTransfersOwnership) {
    MockEngine engine1;
    auto adapter1 = BuildDinoV3Adapter(engine1);
    auto a = PatchEmbedder::Create(std::move(adapter1));
    ASSERT_TRUE(a.has_value());

    MockEngine engine2;
    auto adapter2 = BuildDinoV3Adapter(engine2);
    auto b = PatchEmbedder::Create(std::move(adapter2));
    ASSERT_TRUE(b.has_value());

    auto surface = MakeSurfaceImage();

    // b 移动前可 Extract
    auto result_before = b->Extract(surface);
    ASSERT_FALSE(result_before.has_value());
    EXPECT_EQ(result_before.error().code, ErrorCode::Embedding_NotGpuImage);

    // 移动赋值 a → b
    *b = std::move(*a);

    // a（源）被移走，不能 Extract
    auto result_a = a->Extract(surface);
    ASSERT_FALSE(result_a.has_value());
    EXPECT_EQ(result_a.error().code, ErrorCode::Inference_EngineExecutionFailed);

    // b（目标）仍可 Extract
    auto result_b = b->Extract(surface);
    ASSERT_FALSE(result_b.has_value());
    EXPECT_EQ(result_b.error().code, ErrorCode::Embedding_NotGpuImage);
}

}  // namespace
