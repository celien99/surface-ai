// end_to_end_regression_test.cpp — Full pipeline regression test
//
// Verifies the complete pipeline produces deterministic, repeatable results
// when given known inputs. This serves as a safety net against regressions.
//
// Test data is generated programmatically — no external binary fixtures.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/context.h>
#include <sai/detection/detection_result.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedding.h>
#include <sai/embedding/simple_patch_embedder.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/image/preprocess.h>
#include <sai/io/exporter.h>
#include <sai/io/importer.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/pipeline/pipeline.h>
#include <sai/reasoner/reasoner.h>
#include <sai/reasoner/decision_tree.h>
#include <sai/rule/rule_engine.h>

namespace {

// Generate a simple RGB8 test image (solid color).
auto MakeTestImage(std::size_t w, std::size_t h,
                   std::uint8_t r, std::uint8_t g, std::uint8_t b)
    -> sai::image::RawImage {
    sai::image::ImageMeta meta;
    meta.width = w;
    meta.height = h;
    meta.channels = 3;
    meta.pixel_format = sai::image::PixelFormat::RGB8;

    std::vector<std::uint8_t> pixels(w * h * 3);
    for (std::size_t i = 0; i < pixels.size(); i += 3) {
        pixels[i] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
    }
    return sai::image::RawImage::FromOwnedBuffer(std::move(pixels), meta);
}

}  // anonymous namespace

// ── EndToEndRegression ──────────────────────────────────────────────

class EndToEndRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = std::make_unique<sai::Context>();
        (void)ctx_->Initialize();
        (void)ctx_->Start();
    }

    void TearDown() override {
        (void)ctx_->Stop();
    }

    std::unique_ptr<sai::Context> ctx_;
};

TEST_F(EndToEndRegressionTest, PipelineProducesDeterministicResult) {
    using namespace sai;

    // 1. Build a minimal coreset (2 normal patches, 128-dim) via LoadFromFile.
    constexpr std::size_t kDim = 128;
    constexpr std::size_t kGridH = 4, kGridW = 4;
    constexpr std::size_t kPatches = kGridH * kGridW;  // 16

    // Write two normal vectors to a temp file, then load.
    std::vector<float> normal_patches(2 * kDim, 0.5F);
    auto tmp_path = std::filesystem::temp_directory_path() / "regression_coreset.bin";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(normal_patches.data()),
                static_cast<std::streamsize>(normal_patches.size() * sizeof(float)));
    }
    auto bank_result = detection::FeatureBank::LoadFromFile(tmp_path, kDim);
    ASSERT_TRUE(bank_result.has_value());
    auto bank = std::make_unique<detection::FeatureBank>(std::move(*bank_result));
    std::filesystem::remove(tmp_path);

    // 2. Create a PatchCore with this coreset.
    detection::PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = kDim;
    pc_cfg.image_width = 56;   // 4 * 14
    pc_cfg.image_height = 56;
    pc_cfg.patch_size = 14;
    pc_cfg.k_nearest = 1;
    pc_cfg.anomaly_threshold = 0.5F;
    pc_cfg.enable_adaptive_threshold = false;
    pc_cfg.enable_whitening = false;

    auto patch_core = std::make_shared<detection::PatchCore>(pc_cfg);
    patch_core->SetFeatureBank(std::move(bank));

    // 3. Create a test embedding (all 0.5 = normal, exact match to coreset).
    std::vector<float> normal_embed_data(kPatches * kDim, 0.5F);
    embedding::EmbeddingMeta emb_meta;
    emb_meta.model_name = "test";
    emb_meta.type = embedding::EmbeddingType::Patch;
    emb_meta.dim = kDim;
    emb_meta.count = kPatches;
    emb_meta.grid = {kGridH, kGridW};
    auto normal_emb = embedding::Embedding::FromCpu(
        std::move(normal_embed_data), emb_meta);

    // 4. Run PatchCore::Detect → should be normal (all zero anomaly scores).
    auto detect_result = patch_core->Detect(normal_emb);
    ASSERT_TRUE(detect_result.has_value());

    auto& dr = *detect_result;
    EXPECT_FLOAT_EQ(dr.image_level_score, 0.0F);
    EXPECT_FALSE(dr.IsDefective(0.5F));
    EXPECT_TRUE(dr.regions.empty());

    // 5. Create an anomalous embedding (last patch is far away).
    std::vector<float> anomalous_data(kPatches * kDim, 0.5F);
    // Make the last patch very different
    for (std::size_t d = 0; d < kDim; ++d) {
        anomalous_data[(kPatches - 1) * kDim + d] = 100.0F;
    }
    auto anomalous_emb = embedding::Embedding::FromCpu(
        std::move(anomalous_data), emb_meta);

    auto anom_result = patch_core->Detect(anomalous_emb);
    ASSERT_TRUE(anom_result.has_value());
    auto& adr = *anom_result;
    EXPECT_GT(adr.image_level_score, 0.5F);
    EXPECT_TRUE(adr.IsDefective(0.5F));
    EXPECT_FALSE(adr.regions.empty());
}

TEST_F(EndToEndRegressionTest, EmbedderProducesExpectedGridShape) {
    using namespace sai;

    // Create a SimplePatchEmbedder and verify output shape.
    embedding::SimplePatchEmbedderConfig sp_cfg;
    sp_cfg.image_width = 56;
    sp_cfg.image_height = 56;
    sp_cfg.patch_size = 14;
    sp_cfg.feature_dim = 64;

    auto emb_result = embedding::SimplePatchEmbedder::Create(sp_cfg);
    ASSERT_TRUE(emb_result.has_value());
    auto embedder = std::make_unique<embedding::SimplePatchEmbedder>(
        std::move(*emb_result));

    auto img = MakeTestImage(56, 56, 128, 64, 32);
    auto extract_result = embedder->Extract(img);
    ASSERT_TRUE(extract_result.has_value());

    auto& emb = *extract_result;
    EXPECT_EQ(emb.Meta().dim, 64u);
    EXPECT_EQ(emb.Meta().count, 16u);  // (56/14) * (56/14) = 4 * 4 = 16
    EXPECT_EQ(emb.Meta().grid[0], 4u);
    EXPECT_EQ(emb.Meta().grid[1], 4u);
    EXPECT_EQ(emb.Meta().model_name, "SimplePatch");
    EXPECT_EQ(emb.SizeBytes(), 16u * 64u * sizeof(float));
}

TEST_F(EndToEndRegressionTest, FeatureBankSaveLoadRoundTrip) {
    using namespace sai;

    constexpr std::size_t kDim = 64;
    std::vector<float> data(100 * kDim, 1.0F);
    for (std::size_t i = 0; i < 100; ++i) {
        data[i * kDim] = static_cast<float>(i);  // make each vector distinct
    }

    // Write data to temp file, load via LoadFromFile (public API).
    auto tmp_path = std::filesystem::temp_directory_path() / "test_coreset2.bin";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    auto bank_result = detection::FeatureBank::LoadFromFile(tmp_path, kDim);
    ASSERT_TRUE(bank_result.has_value());
    EXPECT_EQ(bank_result->NumSamples(), 100u);
    EXPECT_EQ(bank_result->Dim(), 64u);

    // Save to another temp file (round-trip)
    auto tmp_path2 = std::filesystem::temp_directory_path() / "test_coreset2_out.bin";
    auto save_result = bank_result->SaveToFile(tmp_path2);
    ASSERT_TRUE(save_result.has_value());

    // Reload and verify
    auto load_result = detection::FeatureBank::LoadFromFile(tmp_path2, kDim);
    ASSERT_TRUE(load_result.has_value());
    EXPECT_EQ(load_result->NumSamples(), 100u);
    EXPECT_EQ(load_result->Dim(), 64u);

    // Search for an exact match → distance should be ~0
    auto dists = load_result->Search(data.data(), 1, 1);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_LT(dists[0], 1e-5F);

    // Cleanup
    std::filesystem::remove(tmp_path);
    std::filesystem::remove(tmp_path2);
}
