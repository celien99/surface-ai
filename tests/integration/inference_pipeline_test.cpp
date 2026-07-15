// inference_pipeline_test.cpp — Task 9: M3 推理管线端到端集成测试
// MockEngine → DinoV3Adapter → PatchEmbedder → PatchCore → DetectionResult
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/context.h>
#include <sai/detection/detection_result.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedder.h>
#include <sai/embedding/embedding.h>
#include <sai/image/image.h>
#include <sai/image/surface_image.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/inference/inference_engine.h>
#include <sai/inference/mock_engine.h>

namespace {

namespace fs = std::filesystem;

using namespace sai::inference;
using namespace sai::embedding;
using namespace sai::detection;

// ============================================================================
// Helpers
// ============================================================================

auto WriteFloatMatrix(const fs::path& path, const std::vector<float>& data) -> void {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

// ============================================================================
// Pipeline Fixture: realistic DINOv3 dimensions (518, 14, 1024, grid=37)
// ============================================================================

class InferencePipelineTest : public ::testing::Test {
protected:
    static constexpr std::size_t kImageSize = 518;
    static constexpr std::size_t kPatchSize = 14;
    static constexpr std::size_t kEmbedDim = 1024;
    static constexpr std::size_t kGridSize = kImageSize / kPatchSize;   // 37
    static constexpr std::size_t kPatchCount = kGridSize * kGridSize;   // 1369

    void SetUp() override {
        // coreset: 2 normal prototype vectors (1024-dim each)
        // Vector 0 = all 0.0f  (normal cluster A)
        // Vector 1 = all 1.0f  (normal cluster B)
        std::vector<float> coreset(2 * kEmbedDim, 0.0f);
        for (std::size_t i = kEmbedDim; i < 2 * kEmbedDim; ++i) {
            coreset[i] = 1.0f;
        }

        coreset_path_ = fs::temp_directory_path() / "test_pipeline_coreset.bin";
        WriteFloatMatrix(coreset_path_, coreset);

        PatchCore::Config cfg;
        cfg.feature_bank_path = coreset_path_;
        cfg.image_width = kImageSize;
        cfg.image_height = kImageSize;
        cfg.patch_size = kPatchSize;
        cfg.embed_dim = kEmbedDim;
        cfg.k_nearest = 1;
        cfg.anomaly_threshold = 0.5f;
        cfg.gaussian_sigma = 4;

        patch_core_ = std::make_unique<PatchCore>(cfg);
    }

    void TearDown() override { fs::remove(coreset_path_); }

    // normal: every patch == coreset vector 0 (all zeros)
    auto MakeNormalEmbedding() const -> Embedding {
        EmbeddingMeta meta;
        meta.model_name = "DINOv3";
        meta.type = EmbeddingType::Patch;
        meta.dim = kEmbedDim;
        meta.count = kPatchCount;
        meta.grid = {kGridSize, kGridSize};

        std::vector<float> data(kPatchCount * kEmbedDim, 0.0f);
        return Embedding::FromCpu(std::move(data), std::move(meta));
    }

    // anomalous: 1368 patches == coreset[0]; last patch = far outlier
    auto MakeAnomalousEmbedding() const -> Embedding {
        EmbeddingMeta meta;
        meta.model_name = "DINOv3";
        meta.type = EmbeddingType::Patch;
        meta.dim = kEmbedDim;
        meta.count = kPatchCount;
        meta.grid = {kGridSize, kGridSize};

        std::vector<float> data(kPatchCount * kEmbedDim, 0.0f);
        // last patch: all 100.0f — far from both coreset vectors
        std::size_t offset = (kPatchCount - 1) * kEmbedDim;
        for (std::size_t j = 0; j < kEmbedDim; ++j) {
            data[offset + j] = 100.0f;
        }
        return Embedding::FromCpu(std::move(data), std::move(meta));
    }

    fs::path coreset_path_;
    std::unique_ptr<PatchCore> patch_core_;
    sai::Context ctx_;
};

// ============================================================================
// Test 1: Construction chain — MockEngine → DinoV3Adapter → PatchEmbedder
// ============================================================================

TEST_F(InferencePipelineTest, MockEngineToEmbedderConstruction) {
    MockEngine engine;

    // DINOv3 bindings: 1 input (B, C, H, W), 1 output (B, H_p, W_p, D)
    auto input_shape = std::vector<std::int64_t>{
        1, 3, static_cast<std::int64_t>(kImageSize), static_cast<std::int64_t>(kImageSize)};
    TensorBinding in{"pixel_values", input_shape, 0, nullptr};

    std::vector<float> output_buffer(kPatchCount * kEmbedDim);
    auto output_shape = std::vector<std::int64_t>{
        1, static_cast<std::int64_t>(kGridSize), static_cast<std::int64_t>(kGridSize),
        static_cast<std::int64_t>(kEmbedDim)};
    std::size_t output_bytes = output_buffer.size() * sizeof(float);
    TensorBinding out{"features", output_shape, output_bytes, output_buffer.data()};

    ASSERT_TRUE(engine.Load("dino.engine", {in}, {out}).has_value());

    // Synthetic fill callback: deterministic per-channel pattern
    engine.SetOutputFillCallback([](std::string_view /*name*/, void* ptr, std::size_t sz) {
        auto* f = static_cast<float*>(ptr);
        auto count = sz / sizeof(float);
        for (std::size_t i = 0; i < count; ++i) {
            f[i] = static_cast<float>(i % 1024) * 0.001f;
        }
    });

    // Create adapter
    DinoV3Config cfg{.engine_path = "dino.engine",
                     .image_size = kImageSize,
                     .patch_size = kPatchSize,
                     .embed_dim = kEmbedDim};
    auto adapter = DinoV3Adapter::Create(engine, cfg);
    ASSERT_TRUE(adapter.has_value());
    EXPECT_EQ(adapter->ModelName(), "DINOv3");

    // Create embedder
    auto embedder = PatchEmbedder::Create(std::move(*adapter));
    ASSERT_TRUE(embedder.has_value());
    EXPECT_EQ(embedder->ModelName(), "DINOv3");

    // Extract with CPU image: expect GPU guard rejection
    auto img = sai::image::SurfaceImage::FromOwnedBuffer(
        std::vector<std::uint8_t>(kImageSize * kImageSize * 3, 128),
        sai::image::ImageMeta{.width = kImageSize,
                              .height = kImageSize,
                              .channels = 3,
                              .pixel_format = sai::image::PixelFormat::RGB8});

    auto emb_result = embedder->Extract(img);
    EXPECT_FALSE(emb_result.has_value());
    // Must be Embedding_NotGpuImage (first guard before GPU inference path)
    EXPECT_EQ(emb_result.error().code, sai::ErrorCode::Embedding_NotGpuImage);
}

// ============================================================================
// Test 2: Normal image → low anomaly score, no regions, not defective
// ============================================================================

TEST_F(InferencePipelineTest, NormalImageNoAnomaly) {
    ASSERT_TRUE(patch_core_->Initialize(ctx_).has_value());

    auto embedding = MakeNormalEmbedding();

    // Verify embedding metadata
    EXPECT_EQ(embedding.Meta().count, kPatchCount);
    EXPECT_EQ(embedding.Meta().dim, kEmbedDim);
    EXPECT_EQ(embedding.Meta().grid[0], kGridSize);
    EXPECT_EQ(embedding.Meta().grid[1], kGridSize);
    EXPECT_EQ(embedding.Meta().type, EmbeddingType::Patch);
    EXPECT_FALSE(embedding.IsOnGpu());

    auto result = patch_core_->Detect(embedding);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto& det = result.value();

    // AnomalyMap dimensions
    EXPECT_EQ(det.anomaly_map.grid_h, kGridSize);
    EXPECT_EQ(det.anomaly_map.grid_w, kGridSize);
    EXPECT_EQ(det.anomaly_map.scores.size(), kPatchCount);

    // All scores are 0 (exact match with coreset)
    for (auto s : det.anomaly_map.scores) {
        EXPECT_FLOAT_EQ(s, 0.0f);
    }

    // Image-level score = 0 → below threshold
    EXPECT_FLOAT_EQ(det.image_level_score, 0.0f);
    EXPECT_FALSE(det.IsDefective(0.5f));

    // No defect regions
    EXPECT_TRUE(det.regions.empty());
}

// ============================================================================
// Test 3: Anomalous image → high anomaly score, regions present, defective
// ============================================================================

TEST_F(InferencePipelineTest, AnomalousImageDetected) {
    ASSERT_TRUE(patch_core_->Initialize(ctx_).has_value());

    auto embedding = MakeAnomalousEmbedding();
    auto result = patch_core_->Detect(embedding);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto& det = result.value();

    // AnomalyMap dimensions
    EXPECT_EQ(det.anomaly_map.grid_h, kGridSize);
    EXPECT_EQ(det.anomaly_map.grid_w, kGridSize);
    EXPECT_EQ(det.anomaly_map.scores.size(), kPatchCount);

    // Most patches should be 0 (normal); last patch = 1.0 (anomalous)
    std::size_t anomalous_idx = kPatchCount - 1;
    for (std::size_t i = 0; i < kPatchCount; ++i) {
        if (i == anomalous_idx) {
            EXPECT_FLOAT_EQ(det.anomaly_map.scores[i], 1.0f);
        } else {
            EXPECT_FLOAT_EQ(det.anomaly_map.scores[i], 0.0f);
        }
    }

    // Image-level score = max score = 1.0
    EXPECT_FLOAT_EQ(det.image_level_score, 1.0f);

    // IsDefective at various thresholds
    EXPECT_TRUE(det.IsDefective(0.5f));
    EXPECT_TRUE(det.IsDefective(0.99f));
    EXPECT_FALSE(det.IsDefective(1.5f));

    // Defect regions should be present (anomalous patch upsampled → region)
    EXPECT_FALSE(det.regions.empty());

    // Verify region properties
    for (const auto& region : det.regions) {
        EXPECT_GT(region.max_anomaly_score, 0.0f);
        EXPECT_GE(region.mean_anomaly_score, 0.0f);
        EXPECT_GT(region.area_pixels, 0U);
        EXPECT_FALSE(region.bounding_box.IsEmpty());
    }
}

// ============================================================================
// Test 4: DetectBatch with two embeddings (normal + anomalous)
// ============================================================================

TEST_F(InferencePipelineTest, DetectBatchNormalAndAnomalous) {
    ASSERT_TRUE(patch_core_->Initialize(ctx_).has_value());

    auto normal = MakeNormalEmbedding();
    auto anomalous = MakeAnomalousEmbedding();

    std::array<const Embedding*, 2> ptrs = {&normal, &anomalous};
    auto results = patch_core_->DetectBatch(ptrs);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results.value().size(), 2U);

    const auto& r0 = results.value()[0];
    const auto& r1 = results.value()[1];

    // First result (normal)
    EXPECT_FLOAT_EQ(r0.image_level_score, 0.0f);
    EXPECT_FALSE(r0.IsDefective(0.5f));
    EXPECT_TRUE(r0.regions.empty());

    // Second result (anomalous)
    EXPECT_FLOAT_EQ(r1.image_level_score, 1.0f);
    EXPECT_TRUE(r1.IsDefective(0.5f));
    EXPECT_FALSE(r1.regions.empty());
}

// ============================================================================
// Test 5: Detect without Initialize returns error
// ============================================================================

TEST_F(InferencePipelineTest, DetectWithoutInitializeReturnsError) {
    // Use a separate PatchCore — fixture's patch_core_ calls Initialize in some tests
    PatchCore::Config cfg;
    cfg.feature_bank_path = coreset_path_;
    cfg.image_width = kImageSize;
    cfg.image_height = kImageSize;
    cfg.patch_size = kPatchSize;
    cfg.embed_dim = kEmbedDim;

    auto pc = PatchCore(cfg);
    auto embedding = MakeNormalEmbedding();
    auto result = pc.Detect(embedding);
    EXPECT_FALSE(result.has_value());
}

// ============================================================================
// Test 6: Invalid patch grid returns error
// ============================================================================

TEST_F(InferencePipelineTest, InvalidPatchGridReturnsError) {
    ASSERT_TRUE(patch_core_->Initialize(ctx_).has_value());

    // Embedding with wrong grid size
    EmbeddingMeta meta;
    meta.model_name = "DINOv3";
    meta.type = EmbeddingType::Patch;
    meta.dim = kEmbedDim;
    meta.count = 4;
    meta.grid = {2, 2};  // config expects 37×37

    std::vector<float> data(4 * kEmbedDim, 0.0f);
    auto embedding = Embedding::FromCpu(std::move(data), std::move(meta));

    auto result = patch_core_->Detect(embedding);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Detection_InvalidPatchGrid);
}

}  // namespace
