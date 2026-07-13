#include <sai/retrieval/vector_path.h>

#include <cstring>
#include <fstream>

#include <gtest/gtest.h>

#include <sai/detection/feature_bank.h>

namespace sai::retrieval {
namespace {

class VectorPathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a small FeatureBank: 10 samples of dim 4
        std::vector<float> data = {
            0.0F, 0.0F, 0.0F, 0.0F,  //
            1.0F, 1.0F, 1.0F, 1.0F,  //
            2.0F, 2.0F, 2.0F, 2.0F,  //
            3.0F, 3.0F, 3.0F, 3.0F,  //
            4.0F, 4.0F, 4.0F, 4.0F,  //
            5.0F, 5.0F, 5.0F, 5.0F,  //
            6.0F, 6.0F, 6.0F, 6.0F,  //
            7.0F, 7.0F, 7.0F, 7.0F,  //
            8.0F, 8.0F, 8.0F, 8.0F,  //
            9.0F, 9.0F, 9.0F, 9.0F,  //
        };
        auto tmp = std::filesystem::temp_directory_path() / "test_feature_bank.f32";
        {
            std::ofstream out(tmp, std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size() * sizeof(float)));
        }
        auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 4);
        ASSERT_TRUE(bank.has_value());
        bank_ = std::make_unique<sai::detection::FeatureBank>(std::move(*bank));
        std::filesystem::remove(tmp);
        vec_path_ = std::make_unique<VectorPath>(*bank_);
    }

    std::unique_ptr<sai::detection::FeatureBank> bank_;
    std::unique_ptr<VectorPath> vec_path_;
};

TEST_F(VectorPathTest, TopKSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::TopK;
    cfg.k = 3;

    float query[] = {0.1F, 0.1F, 0.1F, 0.1F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 3);
    EXPECT_EQ(results->at(0).index, 0);  // closest to [0,0,0,0]
    EXPECT_LT(results->at(0).distance, results->at(1).distance);
}

TEST_F(VectorPathTest, RangeSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::Range;
    cfg.range_threshold = 10.0F;  // L2 distance threshold for 4-dim vectors near [0,0,0,0]

    float query[] = {0.0F, 0.0F, 0.0F, 0.0F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_GT(results->size(), 0);
    for (const auto& r : *results) {
        EXPECT_LT(r.distance, 10.0F);
    }
}

TEST_F(VectorPathTest, HybridSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::Hybrid;
    cfg.k = 2;
    cfg.id_subset = {5, 6, 7};  // only search among indices 5-7

    float query[] = {5.1F, 5.1F, 5.1F, 5.1F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2);
    EXPECT_EQ(results->at(0).index, 5);  // closest in subset
}

TEST_F(VectorPathTest, EmptyIndexReturnsError) {
    // Load a FeatureBank and move from it, leaving an empty moved-from bank.
    // FeatureBank's default constructor is private, but after a move the source
    // is in a valid-but-unspecified state with null index_ and zero counts,
    // which satisfies the empty-index check in VectorPath::Search.
    auto tmp = std::filesystem::temp_directory_path() / "test_empty.f32";
    {
        std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    auto loaded = sai::detection::FeatureBank::LoadFromFile(tmp, 4);
    std::filesystem::remove(tmp);
    ASSERT_TRUE(loaded.has_value());

    // Move the contents out; *loaded is now in a moved-from (empty) state
    sai::detection::FeatureBank kept = std::move(*loaded);
    VectorPath empty_path(*loaded);

    float query[] = {0.0F, 0.0F, 0.0F, 0.0F};
    VectorPath::Config cfg;
    auto result = empty_path.Search(query, cfg);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Retrieval_EmptyIndex);
}

TEST_F(VectorPathTest, HybridFallbackOnEmptySubset) {
    // Hybrid mode with empty id_subset should fall back to TopK
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::Hybrid;
    cfg.k = 2;
    // id_subset is empty (default)

    float query[] = {0.1F, 0.1F, 0.1F, 0.1F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2);
    EXPECT_EQ(results->at(0).index, 0);  // closest to [0,0,0,0]
}

}  // namespace
}  // namespace sai::retrieval
