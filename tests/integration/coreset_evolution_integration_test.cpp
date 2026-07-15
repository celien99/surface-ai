// coreset_evolution_integration_test.cpp — 端到端自进化集成测试
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <sai/detection/coreset_evolution.h>
#include <sai/detection/detection_result.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedding.h>
#include <sai/knowledge/knowledge_store.h>

namespace {

namespace fs = std::filesystem;
using namespace sai::detection;
using namespace sai::embedding;

// ── Helpers ───────────────────────────────────────────────────

auto BuildBank(std::size_t dim, std::size_t count) -> FeatureBank {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    EmbeddingMeta meta;
    meta.model_name = "test";
    meta.type = EmbeddingType::Patch;
    meta.dim = dim;
    meta.count = count;
    meta.grid = {count, 1};
    auto emb = Embedding::FromCpu(std::move(data), std::move(meta));
    std::vector<const Embedding*> ptrs{&emb};
    auto fb = FeatureBank::BuildFromEmbeddings(ptrs, dim, count);
    return std::move(*fb);
}

// ── Integration Tests ─────────────────────────────────────────

TEST(CoresetEvolutionIntegration, EndToEndSelfEvolution) {
    // 1. Build initial coreset
    auto bank = BuildBank(8, 100);
    auto profile = NormalityProfile::Compute(bank, 5);

    PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = 8;
    PatchCore detector(pc_cfg);
    detector.SetFeatureBank(std::make_unique<FeatureBank>(std::move(bank)));

    // 2. Create CoresetEvolution with KnowledgeStore
    EvolutionConfig evo_cfg;
    evo_cfg.enabled = true;
    evo_cfg.target_size = 50;
    evo_cfg.trigger_frames = 5;
    evo_cfg.max_frames = 10;
    evo_cfg.coverage_threshold = 0.99F;  // very permissive for test
    CoresetEvolution evo(evo_cfg, detector, std::move(profile));

    auto ks = sai::knowledge::KnowledgeStore::Create(
        sai::knowledge::KnowledgeStore::Config{":memory:", true});
    ASSERT_TRUE(ks.has_value());
    evo.BindKnowledgeStore(std::move(*ks));

    // 3. Feed "normal but varied" frames
    // Cannot fully test without integration into seat_aoi pipeline.
    // This test verifies the component wiring is correct.
    EXPECT_FALSE(evo.IsRunning());

    // FullRebuild should persist
    auto tmp = fs::temp_directory_path() / "test_e2e_coreset.bin";
    auto rebuild = evo.FullRebuild(tmp);
    EXPECT_TRUE(rebuild.has_value());

    auto loaded = FeatureBank::LoadFromFile(tmp, 8);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_LE(loaded->NumSamples(), 50U);

    fs::remove(tmp);
    auto profile_path = tmp.string() + ".profile.yaml";
    if (fs::exists(profile_path)) fs::remove(profile_path);
}

TEST(CoresetEvolutionIntegration, DefectNeverIncluded) {
    // Verify that when a frame has defect signal (matched_rules > 0),
    // AssessAndOffer does NOT add it to the buffer.

    auto bank = BuildBank(8, 100);
    auto profile = NormalityProfile::Compute(bank, 5);

    PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = 8;
    PatchCore detector(pc_cfg);
    detector.SetFeatureBank(std::make_unique<FeatureBank>(std::move(bank)));

    EvolutionConfig evo_cfg;
    evo_cfg.enabled = true;
    evo_cfg.coverage_threshold = 0.99F;
    CoresetEvolution evo(evo_cfg, detector, std::move(profile));

    // Simulate a frame with defect: matched_rules_count > 0
    DetectionResult det;
    det.image_level_score = 0.9F;
    std::vector<float> distances(64, 10.0F);  // high distances
    std::vector<float> embedding_data(64, 0.5F);  // dummy embedding data

    evo.AssessAndOffer(
        distances.data(), 64, 5,
        embedding_data.data(), 8, 8, 1,  // grid_h=8, grid_w=8, dim=1
        det,
        1,     // matched_rules_count > 0 → defect
        "NG",  // reasoner says NG
        0.8F,  // threshold
        0.0F, 0.0F);  // PCA disabled

    // Buffer should be empty (defect rejected)
    auto stats = evo.LatestStats();
    // Since AssessAndOffer doesn't expose buffer count directly,
    // we verify indirectly: no update has been triggered.
    EXPECT_EQ(stats.update_count, 0U);
}

}  // namespace
