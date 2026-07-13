#include <sai/retrieval/hybrid_retriever.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/detection/feature_bank.h>
#include <sqlite3.h>
#include <gtest/gtest.h>
#include <fstream>

namespace sai::retrieval {
namespace {

class HybridRetrieverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup SQLite with nodes
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M1"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M2"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M3"}');
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);

        // Setup FeatureBank with 3 vectors corresponding to the 3 nodes
        std::vector<float> data = {
            0.0F, 0.0F,   // M1
            3.0F, 3.0F,   // M2
            6.0F, 6.0F,   // M3
        };
        auto tmp = std::filesystem::temp_directory_path() / "test_hr.f32";
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
        out.close();
        auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 2);
        std::filesystem::remove(tmp);
        ASSERT_TRUE(bank.has_value());
        bank_ = std::make_unique<sai::detection::FeatureBank>(std::move(*bank));
    }
    void TearDown() override { sqlite3_close(db_); }

    sqlite3* db_ = nullptr;
    std::unique_ptr<sai::detection::FeatureBank> bank_;
};

TEST_F(HybridRetrieverTest, RetrieveWithWeightedFusion) {
    VectorPath vec_path(*bank_);
    MetadataPath meta_path(db_);
    auto fusion = std::make_unique<WeightedFusion>();

    HybridRetriever retriever(vec_path, meta_path, std::move(fusion));

    HybridRetriever::Config cfg;
    cfg.vector.mode = VectorPath::Mode::TopK;
    cfg.vector.k = 3;

    float query[] = {0.1F, 0.1F};  // closest to M1
    std::vector<std::int64_t> vec_to_node = {1, 2, 3};  // vec index → node_id

    auto results = retriever.Retrieve(query, cfg, vec_to_node);
    ASSERT_TRUE(results.has_value());
    ASSERT_GE(results->size(), 1);
    EXPECT_EQ(results->at(0).node_id, 1);  // M1 closest to query
    EXPECT_GT(results->at(0).scores.fused_score, 0.0F);
}

TEST_F(HybridRetrieverTest, SetFusionStrategy) {
    VectorPath vec_path(*bank_);
    MetadataPath meta_path(db_);

    HybridRetriever retriever(vec_path, meta_path, std::make_unique<WeightedFusion>());
    retriever.SetFusion(std::make_unique<RRFFusion>());

    HybridRetriever::Config cfg;
    cfg.vector.mode = VectorPath::Mode::TopK;
    cfg.vector.k = 2;

    float query[] = {0.0F, 0.0F};
    std::vector<std::int64_t> vec_to_node = {1, 2, 3};

    auto results = retriever.Retrieve(query, cfg, vec_to_node);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->at(0).scores.fusion_strategy, "RRFFusion");
}

}  // namespace
}  // namespace sai::retrieval
