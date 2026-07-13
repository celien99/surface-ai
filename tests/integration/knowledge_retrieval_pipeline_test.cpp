// knowledge_retrieval_pipeline_test.cpp — M4 验证点：端到端知识写入 + 检索
#include <sai/knowledge/knowledge_store.h>
#include <sai/retrieval/hybrid_retriever.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/detection/feature_bank.h>
#include <gtest/gtest.h>
#include <fstream>

namespace {

TEST(KnowledgeRetrievalPipelineTest, WriteAndRetrieveWithFusedScore) {
    // 1. Create KnowledgeStore with :memory:
    sai::knowledge::KnowledgeStore::Config kcfg;
    kcfg.db_path = ":memory:";
    auto store = sai::knowledge::KnowledgeStore::Create(kcfg);
    ASSERT_TRUE(store.has_value());

    // 2. Insert Material node with properties
    sai::knowledge::KnowledgeRecord mat_props;
    mat_props.fields["material_code"] = std::string("LEATHER-001");
    mat_props.fields["name"] = std::string("Nappa 真皮");
    auto mat_id = (*store)->InsertNode("Material", mat_props, "importer");
    ASSERT_TRUE(mat_id.has_value());

    // 3. Insert Supplier node
    sai::knowledge::KnowledgeRecord sup_props;
    sup_props.fields["supplier_code"] = std::string("SUP-2026");
    sup_props.fields["name"] = std::string("某供应商");
    auto sup_id = (*store)->InsertNode("Supplier", sup_props, "importer");
    ASSERT_TRUE(sup_id.has_value());

    // 4. Create edge: Material -[supplied_by]-> Supplier
    sai::knowledge::KnowledgeRecord edge_props;
    auto edge_id = (*store)->InsertEdge(*mat_id, *sup_id, "supplied_by", edge_props);
    ASSERT_TRUE(edge_id.has_value());

    // 5. Verify graph traversal
    auto paths = (*store)->Traverse(*mat_id, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->at(0).targets.size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup_id);

    // 6. Build FeatureBank with CLIP-like embeddings for each node
    // Material LEATHER-001: [0.1, 0.2, 0.3, 0.4]
    // Supplier SUP-2026:    [0.5, 0.6, 0.7, 0.8]
    // (In production these would come from CLIPAdapter)
    std::vector<float> data = {
        0.1F, 0.2F, 0.3F, 0.4F,  // vec[0] = Material LEATHER-001
        0.5F, 0.6F, 0.7F, 0.8F,  // vec[1] = Supplier SUP-2026
    };
    auto tmp = std::filesystem::temp_directory_path() / "test_pipeline.f32";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 4);
    std::filesystem::remove(tmp);
    ASSERT_TRUE(bank.has_value());

    // 7. vec_to_knowledge mapping: vec_index → node_id
    std::vector<std::int64_t> vec_to_node = {*mat_id, *sup_id};

    // 8. Create VectorPath + MetadataPath + WeightedFusion
    sai::retrieval::VectorPath vec_path(*bank);
    sai::retrieval::MetadataPath meta_path((*store)->DbHandle());
    auto fusion = std::make_unique<sai::retrieval::WeightedFusion>();

    // 9. Create HybridRetriever
    sai::retrieval::HybridRetriever retriever(vec_path, meta_path, std::move(fusion));

    sai::retrieval::HybridRetriever::Config cfg;
    cfg.vector.mode = sai::retrieval::VectorPath::Mode::TopK;
    cfg.vector.k = 2;
    cfg.metadata.filters.push_back(sai::retrieval::FilterCondition{
        "material_code", sai::retrieval::FilterOp::Equal, std::string("LEATHER-001")
    });

    // 10. Retrieve with query vector close to LEATHER-001
    float query[] = {0.15F, 0.25F, 0.35F, 0.45F};
    auto results = retriever.Retrieve(query, cfg, vec_to_node);

    // 11. Assertions
    ASSERT_TRUE(results.has_value());
    ASSERT_GE(results->size(), 1);

    // Top result should be the Material node (closest vector + metadata match)
    EXPECT_EQ(results->at(0).node_id, *mat_id);
    EXPECT_GT(results->at(0).scores.fused_score, 0.0F);
    EXPECT_GT(results->at(0).scores.vector_score, 0.0F);
    EXPECT_EQ(results->at(0).scores.fusion_strategy, "WeightedFusion");
}

TEST(KnowledgeRetrievalPipelineTest, EvolutionTracksInsertAndRetrieve) {
    sai::knowledge::KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = true;
    auto store = sai::knowledge::KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    sai::knowledge::KnowledgeRecord props;
    props.fields["code"] = std::string("BATCH-001");
    auto id = (*store)->InsertNode("Batch", props, "mes_importer");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).changed_by, "mes_importer");
    EXPECT_EQ(history->at(0).operation, sai::knowledge::EvolutionOp::Insert);

    auto changes = (*store)->GetChangesSince(
        std::chrono::system_clock::now() - std::chrono::hours(1));
    ASSERT_TRUE(changes.has_value());
    ASSERT_GE(changes->size(), 1);
}

}  // namespace
