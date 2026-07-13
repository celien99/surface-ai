#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

namespace sai::retrieval {
namespace {

TEST(WeightedFusionTest, BasicFusion) {
    WeightedFusion fusion;
    YAML::Node params;
    params["alpha"] = 0.7F;
    ASSERT_TRUE(fusion.Configure(params).has_value());

    // Vector results: 3 candidates with descending distance
    std::vector<VectorResult> vec_results = {
        {0, 0.1F}, {1, 0.5F}, {2, 2.0F}
    };
    std::vector<std::int64_t> vec_node_ids = {100, 101, 102};

    // Metadata results: only node 101 matched
    std::vector<MetadataResult> meta_results = {
        {101, 1.0F}
    };

    auto fused = fusion.Fuse(vec_results, vec_node_ids, meta_results);
    ASSERT_GE(fused.size(), 1);
    // node 101 should rank high: has both vector and metadata match
    EXPECT_EQ(fused[0].first, 101);
    EXPECT_GT(fused[0].second.fused_score, 0.0F);
    EXPECT_EQ(fused[0].second.fusion_strategy, "WeightedFusion");
}

TEST(RRFFusionTest, BasicFusion) {
    RRFFusion fusion;
    YAML::Node params;
    params["k"] = 60.0F;
    ASSERT_TRUE(fusion.Configure(params).has_value());

    std::vector<VectorResult> vec_results = {
        {0, 0.1F}, {1, 0.5F}, {2, 2.0F}
    };
    std::vector<std::int64_t> vec_node_ids = {100, 101, 102};
    std::vector<MetadataResult> meta_results = {{101, 1.0F}};

    auto fused = fusion.Fuse(vec_results, vec_node_ids, meta_results);
    ASSERT_GE(fused.size(), 1);
    EXPECT_EQ(fused[0].second.fusion_strategy, "RRFFusion");
}

TEST(WeightedFusionTest, ConfigureInvalidAlpha) {
    WeightedFusion fusion;
    YAML::Node params;
    params["alpha"] = 2.0F;  // out of [0,1] range
    auto result = fusion.Configure(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Retrieval_FusionConfigInvalid);
}

}  // namespace
}  // namespace sai::retrieval
