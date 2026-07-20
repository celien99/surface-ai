#include <sai/inference/inference_engine.h>
#include <sai/inference/mock_engine.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/inference/sam2_adapter.h>
#include <sai/inference/clip_adapter.h>

#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace sai::inference;

// ============================================================================
// MockEngine tests
// ============================================================================

TEST(MockEngine, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<MockEngine>);
    static_assert(!std::is_copy_assignable_v<MockEngine>);
    // MockEngine inherits from Object which deletes move; MockEngine must
    // be usable after construction — we don't test move because Object
    // intentionally forbids it for identity-semantics types.
}

TEST(MockEngine, LoadRecordsBindings) {
    MockEngine engine;
    auto inputs = std::vector<TensorBinding>{
        {"image", {1, 3, 518, 518}, 0, nullptr},
    };
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 256, 256, 1024}, 0, nullptr},
    };

    ASSERT_TRUE(engine.Load("dummy.engine", inputs, outputs).has_value());

    EXPECT_EQ(engine.InputBindings().size(), 1U);
    EXPECT_EQ(engine.InputBindings()[0].name, "image");
    EXPECT_EQ(engine.OutputBindings().size(), 1U);
    EXPECT_EQ(engine.OutputBindings()[0].name, "features");
    EXPECT_EQ(engine.OutputBindings()[0].shape.size(), 4U);
    EXPECT_EQ(engine.OutputBindings()[0].shape[3], 1024);
}

TEST(MockEngine, LoadWithoutAnyBindings) {
    MockEngine engine;
    ASSERT_TRUE(engine.Load("dummy.engine", {}, {}).has_value());
    EXPECT_TRUE(engine.InputBindings().empty());
    EXPECT_TRUE(engine.OutputBindings().empty());
}

TEST(MockEngine, InferSucceedsWhenOutputsHaveValidDevicePointers) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, buf.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());
    ASSERT_TRUE(engine.Infer().has_value());
}

TEST(MockEngine, InferReturnsErrorWhenOutputBindingHasNullDevicePtr) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, nullptr}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());
    auto result = engine.Infer();
    EXPECT_FALSE(result.has_value());
}

TEST(MockEngine, SetTensorAddressUpdatesOutputPtr) {
    MockEngine engine;
    std::array<float, 4> buf1{};
    std::array<float, 4> buf2{};
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, buf1.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());

    ASSERT_TRUE(engine.SetTensorAddress("features", buf2.data()).has_value());
    // The binding ptr should now be buf2.data()
    ASSERT_TRUE(engine.Infer().has_value());
}

TEST(MockEngine, SetTensorAddressReturnsErrorForUnknownBinding) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, nullptr}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());

    auto result = engine.SetTensorAddress("nonexistent", nullptr);
    EXPECT_FALSE(result.has_value());
}

TEST(MockEngine, LoadAndInferInvokesCallback) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, buf.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());

    bool called = false;
    engine.SetOutputFillCallback([&](std::string_view name, void* ptr, std::size_t sz) {
        EXPECT_EQ(name, "features");
        EXPECT_EQ(sz, 16U);
        std::memset(ptr, 0xAB, sz);
        called = true;
    });
    ASSERT_TRUE(engine.Infer().has_value());
    EXPECT_TRUE(called);
    // Verify 0xABABABAB pattern was written
    EXPECT_NE(buf[0], 0.0F);
    EXPECT_EQ(std::memcmp(buf.data(), buf.data() + 1, 4), 0);
}

TEST(MockEngine, InferAsyncInvokesCallback) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, buf.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());

    bool called = false;
    engine.SetOutputFillCallback([&](std::string_view name, void* ptr, std::size_t sz) {
        std::memset(ptr, 0xCD, sz);
        called = true;
    });
    ASSERT_TRUE(engine.InferAsync(nullptr).has_value());
    EXPECT_TRUE(called);
}

TEST(MockEngine, InferWithoutCallbackSucceeds) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, buf.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());
    // No callback set — should succeed without calling anything
    ASSERT_TRUE(engine.Infer().has_value());
}

TEST(MockEngine, InferBeforeLoadReturnsError) {
    MockEngine engine;
    auto result = engine.Infer();
    EXPECT_FALSE(result.has_value());
}

TEST(MockEngine, ReloadSucceeds) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{{"features", {1, 4}, 16, nullptr}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());
    ASSERT_TRUE(engine.Reload("new.engine").has_value());
    // After reload, bindings should still be present
    EXPECT_EQ(engine.OutputBindings().size(), 1U);
    EXPECT_EQ(engine.OutputBindings()[0].name, "features");
}

TEST(MockEngine, MultipleInputAndOutputBindings) {
    MockEngine engine;
    auto inputs = std::vector<TensorBinding>{
        {"image", {1, 3, 518, 518}, 0, nullptr},
        {"prompt", {1, 3, 256, 256}, 0, nullptr},
    };
    std::array<float, 16> features_buf{};
    std::array<float, 8> scores_buf{};
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 16}, 64, features_buf.data()},
        {"scores", {1, 8}, 32, scores_buf.data()},
    };
    ASSERT_TRUE(engine.Load("multi.engine", inputs, outputs).has_value());

    int call_count = 0;
    engine.SetOutputFillCallback([&](std::string_view name, void* ptr, std::size_t sz) {
        ++call_count;
        std::memset(ptr, 0, sz);
    });
    ASSERT_TRUE(engine.Infer().has_value());
    EXPECT_EQ(call_count, 2);
}

TEST(MockEngine, InferReturnsErrorIfAnyOutputBindingNullAfterLoad) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 4}, 16, buf.data()},
        {"scores", {1, 2}, 8, nullptr},  // this one is null
    };
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());
    auto result = engine.Infer();
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// DinoV3Adapter tests
// ============================================================================

TEST(DinoV3Adapter, CreateReturnsErrorWhenEngineHasNoOutputBindings) {
    MockEngine engine;
    ASSERT_TRUE(engine.Load("test.engine", {}, {}).has_value());

    DinoV3Config cfg{.engine_path = "test.engine",
                     .image_size = 518,
                     .patch_size = 14,
                     .embed_dim = 1024};
    auto adapter = DinoV3Adapter::Create(engine, cfg);
    EXPECT_FALSE(adapter.has_value());
}

TEST(DinoV3Adapter, CreateSucceedsWhenBindingsMatchConfig) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{
        {"last_hidden_state", {1, 37, 37, 1024}, 0, nullptr},
    };
    ASSERT_TRUE(engine.Load("dino.engine", {}, outputs).has_value());

    DinoV3Config cfg{.engine_path = "dino.engine",
                     .image_size = 518,
                     .patch_size = 14,
                     .embed_dim = 1024};
    auto adapter = DinoV3Adapter::Create(engine, cfg);
    ASSERT_TRUE(adapter.has_value());
    EXPECT_EQ(adapter->ModelName(), "DINOv2");
}

TEST(DinoV3Adapter, CreateFailsWhenEmbedDimMismatch) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{
        {"last_hidden_state", {1, 37, 37, 512}, 0, nullptr},
    };
    ASSERT_TRUE(engine.Load("dino.engine", {}, outputs).has_value());

    DinoV3Config cfg{.engine_path = "dino.engine",
                     .image_size = 518,
                     .patch_size = 14,
                     .embed_dim = 1024};  // mismatch: engine says 512
    auto adapter = DinoV3Adapter::Create(engine, cfg);
    EXPECT_FALSE(adapter.has_value());
}

TEST(DinoV3Adapter, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<DinoV3Adapter>);
    static_assert(!std::is_copy_assignable_v<DinoV3Adapter>);
    static_assert(std::is_move_constructible_v<DinoV3Adapter>);
    static_assert(std::is_move_assignable_v<DinoV3Adapter>);
}

// ============================================================================
// Sam2Adapter tests
// ============================================================================

TEST(Sam2Adapter, CreateReturnsErrorWhenEngineHasNoBindings) {
    MockEngine engine;
    ASSERT_TRUE(engine.Load("test.engine", {}, {}).has_value());

    Sam2Config cfg{.engine_path = "sam2.engine", .image_size = 1024};
    auto adapter = Sam2Adapter::Create(engine, cfg);
    EXPECT_FALSE(adapter.has_value());
}

TEST(Sam2Adapter, CreateSucceedsWhenBindingsMatchConfig) {
    MockEngine engine;
    auto inputs = std::vector<TensorBinding>{
        {"image", {1, 3, 1024, 1024}, 0, nullptr},
        {"prompt", {1, 3, 256, 256}, 0, nullptr},
    };
    auto outputs = std::vector<TensorBinding>{
        {"masks", {1, 1024, 1024}, 0, nullptr},
    };
    ASSERT_TRUE(engine.Load("sam2.engine", inputs, outputs).has_value());

    Sam2Config cfg{.engine_path = "sam2.engine", .image_size = 1024};
    auto adapter = Sam2Adapter::Create(engine, cfg);
    ASSERT_TRUE(adapter.has_value());
}

TEST(Sam2Adapter, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<Sam2Adapter>);
    static_assert(!std::is_copy_assignable_v<Sam2Adapter>);
    static_assert(std::is_move_constructible_v<Sam2Adapter>);
    static_assert(std::is_move_assignable_v<Sam2Adapter>);
}

// ============================================================================
// ClipAdapter tests
// ============================================================================

TEST(ClipAdapter, CreateReturnsErrorWhenEngineHasNoOutputBindings) {
    MockEngine engine;
    ASSERT_TRUE(engine.Load("test.engine", {}, {}).has_value());

    ClipConfig cfg{.engine_path = "clip.engine",
                   .image_size = 224,
                   .embed_dim = 512};
    auto adapter = ClipAdapter::Create(engine, cfg);
    EXPECT_FALSE(adapter.has_value());
}

TEST(ClipAdapter, CreateSucceedsWhenBindingsMatchConfig) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 512}, 0, nullptr},
    };
    ASSERT_TRUE(engine.Load("clip.engine", {}, outputs).has_value());

    ClipConfig cfg{.engine_path = "clip.engine",
                   .image_size = 224,
                   .embed_dim = 512};
    auto adapter = ClipAdapter::Create(engine, cfg);
    ASSERT_TRUE(adapter.has_value());
    EXPECT_EQ(adapter->ModelName(), "CLIP");
}

TEST(ClipAdapter, CreateFailsWhenEmbedDimMismatch) {
    MockEngine engine;
    auto outputs = std::vector<TensorBinding>{
        {"features", {1, 256}, 0, nullptr},
    };
    ASSERT_TRUE(engine.Load("clip.engine", {}, outputs).has_value());

    ClipConfig cfg{.engine_path = "clip.engine",
                   .image_size = 224,
                   .embed_dim = 512};  // mismatch
    auto adapter = ClipAdapter::Create(engine, cfg);
    EXPECT_FALSE(adapter.has_value());
}

TEST(ClipAdapter, TypeTraits) {
    static_assert(!std::is_copy_constructible_v<ClipAdapter>);
    static_assert(!std::is_copy_assignable_v<ClipAdapter>);
    static_assert(std::is_move_constructible_v<ClipAdapter>);
    static_assert(std::is_move_assignable_v<ClipAdapter>);
}

}  // namespace
