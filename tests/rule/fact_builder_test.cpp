#include <gtest/gtest.h>
#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "sai/rule/fact_builder.h"
#include "sai/detection/detection_result.h"
#include "sai/detection/feature_bank.h"
#include "sai/knowledge/knowledge_graph.h"
#include "sai/retrieval/vector_path.h"

// -----------------------------------------------------------------------
// Test fixtures
// -----------------------------------------------------------------------

namespace sai::rule {
namespace {

// Sets up a KnowledgeGraph backed by in-memory SQLite with the expected
// schema (the same schema KnowledgeStore creates in production).
struct TestGraph {
    sqlite3* db = nullptr;
    knowledge::KnowledgeGraph kg{nullptr};

    TestGraph() {
        sqlite3_open(":memory:", &db);
        kg = knowledge::KnowledgeGraph(db);
        const char* schema = R"(
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                relationship TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_nodes_type ON nodes(type);
            CREATE INDEX idx_edges_source ON edges(source_id);
            CREATE INDEX idx_edges_target ON edges(target_id);
        )";
        char* err = nullptr;
        sqlite3_exec(db, schema, nullptr, nullptr, &err);
        if (err) {
            sqlite3_free(err);
        }
    }

    ~TestGraph() {
        if (db) sqlite3_close(db);
    }

    TestGraph(const TestGraph&) = delete;
    auto operator=(const TestGraph&) = delete;
};

// Sets up a small FeatureBank + VectorPath backed by generated float data.
// 5 samples of dim 3:
//   sample 0: {0,0,0}, sample 1: {1,1,1}, ..., sample 4: {4,4,4}
struct TestVectorPath {
    std::unique_ptr<detection::FeatureBank> bank;
    std::unique_ptr<retrieval::VectorPath> vp;

    TestVectorPath() {
        std::vector<float> data = {
            0.0F, 0.0F, 0.0F,  //
            1.0F, 1.0F, 1.0F,  //
            2.0F, 2.0F, 2.0F,  //
            3.0F, 3.0F, 3.0F,  //
            4.0F, 4.0F, 4.0F,  //
        };
        auto tmp = std::filesystem::temp_directory_path() / "fact_builder_fb.f32";
        {
            std::ofstream out(tmp, std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size() * sizeof(float)));
        }
        auto loaded = detection::FeatureBank::LoadFromFile(tmp, 3);
        std::filesystem::remove(tmp);
        if (loaded) {
            bank = std::make_unique<detection::FeatureBank>(std::move(*loaded));
            vp = std::make_unique<retrieval::VectorPath>(*bank);
        }
    }

    explicit operator bool() const noexcept { return vp != nullptr; }
};

// -----------------------------------------------------------------------
// Helper: build a minimal DetectionResult with known values
// -----------------------------------------------------------------------

auto MakeDetection(float image_level, std::size_t n_regions = 1)
    -> detection::DetectionResult
{
    detection::DetectionResult dr;
    dr.image_level_score = image_level;
    dr.inference_latency = std::chrono::nanoseconds(1500000);
    dr.anomaly_map.grid_h = 2;
    dr.anomaly_map.grid_w = 3;
    dr.anomaly_map.scores = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};

    for (std::size_t i = 0; i < n_regions; ++i) {
        auto rx = 10U + static_cast<unsigned>(i * 5);
        auto ry = 20U + static_cast<unsigned>(i * 5);
        detection::RegionProposal rp{
            core::Rect{rx, ry, 100U, 200U},
            0.9F - static_cast<float>(i) * 0.1F,
            0.5F - static_cast<float>(i) * 0.1F,
            500U + i * 100};
        dr.regions.push_back(rp);
    }
    return dr;
}

// ── MapDetection through Build ──────────────────────────────────────

TEST(FactBuilder_Build, MapsDetectionFieldsCorrectly) {
    auto dr = MakeDetection(0.85F, 2);
    FactBuilder fb(nullptr, nullptr);

    auto result = fb.Build("surface-001", dr, {});
    ASSERT_TRUE(result.has_value());

    auto& fb_out = *result;

    // Check scalar fields
    auto score = fb_out.Get("detection.image_level_score");
    ASSERT_TRUE(score.has_value());
    EXPECT_NEAR(score->AsDouble().value(), 0.85, 1e-6);

    auto latency = fb_out.Get("detection.latency_ns");
    ASSERT_TRUE(latency.has_value());
    EXPECT_DOUBLE_EQ(latency->AsDouble().value(), 1500000.0);

    auto max_anom = fb_out.Get("detection.anomaly_map.max_score");
    ASSERT_TRUE(max_anom.has_value());
    EXPECT_NEAR(max_anom->AsDouble().value(), 0.6, 1e-6);  // max of {0.1..0.6}

    auto grid_h = fb_out.Get("detection.grid_h");
    ASSERT_TRUE(grid_h.has_value());
    EXPECT_DOUBLE_EQ(grid_h->AsDouble().value(), 2.0);

    auto grid_w = fb_out.Get("detection.grid_w");
    ASSERT_TRUE(grid_w.has_value());
    EXPECT_DOUBLE_EQ(grid_w->AsDouble().value(), 3.0);

    auto region_count = fb_out.Get("detection.region_count");
    ASSERT_TRUE(region_count.has_value());
    EXPECT_DOUBLE_EQ(region_count->AsDouble().value(), 2.0);

    // Check first region fields
    auto r0_max = fb_out.Get("detection.region.0.max_score");
    ASSERT_TRUE(r0_max.has_value());
    EXPECT_NEAR(r0_max->AsDouble().value(), 0.9, 1e-6);

    auto r0_mean = fb_out.Get("detection.region.0.mean_score");
    ASSERT_TRUE(r0_mean.has_value());
    EXPECT_DOUBLE_EQ(r0_mean->AsDouble().value(), 0.5);

    auto r0_area = fb_out.Get("detection.region.0.area_pixels");
    ASSERT_TRUE(r0_area.has_value());
    EXPECT_DOUBLE_EQ(r0_area->AsDouble().value(), 500.0);

    auto r0_x = fb_out.Get("detection.region.0.x");
    ASSERT_TRUE(r0_x.has_value());
    EXPECT_DOUBLE_EQ(r0_x->AsDouble().value(), 10.0);

    auto r0_y = fb_out.Get("detection.region.0.y");
    ASSERT_TRUE(r0_y.has_value());
    EXPECT_DOUBLE_EQ(r0_y->AsDouble().value(), 20.0);

    auto r0_w = fb_out.Get("detection.region.0.width");
    ASSERT_TRUE(r0_w.has_value());
    EXPECT_DOUBLE_EQ(r0_w->AsDouble().value(), 100.0);

    auto r0_h = fb_out.Get("detection.region.0.height");
    ASSERT_TRUE(r0_h.has_value());
    EXPECT_DOUBLE_EQ(r0_h->AsDouble().value(), 200.0);

    // Check surface.id
    auto sid = fb_out.Get("surface.id");
    ASSERT_TRUE(sid.has_value());
    ASSERT_TRUE(sid->AsString().has_value());
    EXPECT_EQ(*sid->AsString(), "surface-001");

    // Verify source tracking
    auto& src = fb_out.SourceOf("detection.image_level_score");
    EXPECT_EQ(src.kind, FactSourceKind::Direct);
}

TEST(FactBuilder_Build, MapsEmptyRegions) {
    auto dr = MakeDetection(0.5F, 0);  // zero regions
    FactBuilder fb(nullptr, nullptr);

    auto result = fb.Build("empty", dr, {});
    ASSERT_TRUE(result.has_value());

    // region_count should be 0
    auto rc = result->Get("detection.region_count");
    ASSERT_TRUE(rc.has_value());
    EXPECT_DOUBLE_EQ(rc->AsDouble().value(), 0.0);

    // region fields should NOT exist
    EXPECT_FALSE(result->Has("detection.region.0.max_score"));
    EXPECT_FALSE(result->Has("detection.region.0.x"));
}

// ── ResolveGraphPaths ───────────────────────────────────────────────

TEST(FactBuilder_ResolveGraphPaths, SimplePropertyExtraction) {
    TestGraph tg;

    // Insert a node of type "DefectType" with properties
    knowledge::KnowledgeRecord props;
    props.fields["code"] = std::string("SCRATCH");
    props.fields["severity"] = std::int64_t(3);
    auto id = tg.kg.InsertNode("DefectType", props);
    ASSERT_TRUE(id.has_value());

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    FactBase fact;
    auto result = fb.Build("test", MakeDetection(0.5F), {"DefectType->code"});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Verify the property was extracted
    // Key format: "graph.DefectType.code"
    auto val = result->Get("graph.DefectType.code");
    ASSERT_TRUE(val.has_value());
    ASSERT_TRUE(val->AsString().has_value());
    EXPECT_EQ(*val->AsString(), "SCRATCH");
}

TEST(FactBuilder_ResolveGraphPaths, TraversalExtraction) {
    TestGraph tg;

    // Create: Material --supplied_by--> Supplier
    knowledge::KnowledgeRecord mat_props;
    mat_props.fields["name"] = std::string("Leather-A");
    auto mat_id = tg.kg.InsertNode("Material", mat_props);
    ASSERT_TRUE(mat_id.has_value());

    knowledge::KnowledgeRecord sup_props;
    sup_props.fields["reject_rate"] = 0.023;
    auto sup_id = tg.kg.InsertNode("Supplier", sup_props);
    ASSERT_TRUE(sup_id.has_value());

    auto edge_id = tg.kg.InsertEdge(*mat_id, *sup_id, "supplied_by", {});
    ASSERT_TRUE(edge_id.has_value());

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    auto result = fb.Build("test", MakeDetection(0.5F),
                           {"Material->supplied_by.reject_rate"});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Key format: "graph.Material.supplied_by.reject_rate"
    auto val = result->Get("graph.Material.supplied_by.reject_rate");
    ASSERT_TRUE(val.has_value());
    ASSERT_TRUE(val->AsDouble().has_value());
    EXPECT_DOUBLE_EQ(val->AsDouble().value(), 0.023);
}

TEST(FactBuilder_ResolveGraphPaths, MultiHopTraversal) {
    TestGraph tg;

    // Create: Defect --has_cause--> CauseGroup --belongs_to--> Category
    knowledge::KnowledgeRecord def_props;
    def_props.fields["name"] = std::string("Scratch");
    auto def_id = tg.kg.InsertNode("Defect", def_props);
    ASSERT_TRUE(def_id.has_value());

    knowledge::KnowledgeRecord cause_props;
    cause_props.fields["group_name"] = std::string("Mechanical");
    auto cause_id = tg.kg.InsertNode("CauseGroup", cause_props);
    ASSERT_TRUE(cause_id.has_value());

    knowledge::KnowledgeRecord cat_props;
    cat_props.fields["score_multiplier"] = 1.5;
    auto cat_id = tg.kg.InsertNode("Category", cat_props);
    ASSERT_TRUE(cat_id.has_value());

    tg.kg.InsertEdge(*def_id, *cause_id, "has_cause", {});
    tg.kg.InsertEdge(*cause_id, *cat_id, "belongs_to", {});

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    auto result = fb.Build("test", MakeDetection(0.5F),
                           {"Defect->has_cause->belongs_to.score_multiplier"});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto val = result->Get("graph.Defect.has_cause.belongs_to.score_multiplier");
    ASSERT_TRUE(val.has_value());
    ASSERT_TRUE(val->AsDouble().has_value());
    EXPECT_DOUBLE_EQ(val->AsDouble().value(), 1.5);
}

TEST(FactBuilder_ResolveGraphPaths, InvalidPath) {
    TestGraph tg;

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    // Single segment (no "->") should fail
    auto result = fb.Build("test", MakeDetection(0.5F), {"JustOneSegment"});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_InvalidPath);
}

TEST(FactBuilder_ResolveGraphPaths, NoMatchingNodes) {
    TestGraph tg;

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    // No nodes of type "NonExistent" — should be a no-op, not an error
    auto result = fb.Build("test", MakeDetection(0.5F),
                           {"NonExistent->some_property"});
    ASSERT_TRUE(result.has_value());

    // Property should not be set
    EXPECT_FALSE(result->Has("graph.NonExistent.some_property"));
}

// ── RunVectorRetrieval ──────────────────────────────────────────────

TEST(FactBuilder_RunVectorRetrieval, TopKReturnsClosest) {
    TestVectorPath tvp;
    ASSERT_TRUE(static_cast<bool>(tvp))
        << "Could not set up FeatureBank/VectorPath";

    // Build FactBuilder with a VectorPath
    auto vp_shared = std::make_shared<retrieval::VectorPath>(*tvp.vp);
    FactBuilder fb(nullptr, vp_shared);

    // Create a FactBase with a query vector under "embedding.test"
    FactBase fact;
    fact.Set("embedding.test",
             Value::OfList({Value::Of(0.1), Value::Of(0.1), Value::Of(0.1)}),
             FactSource{FactSourceKind::Direct, "test vector"});

    auto result = fb.RunVectorRetrieval(fact, "embedding.test");
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Should have retrieval.count = 10 (default k) but only 5 samples in bank
    // FeatureBank has 5 samples, so we should get min(5, 10) = 5 results
    auto count = fact.Get("retrieval.count");
    ASSERT_TRUE(count.has_value());
    auto cnt_d = count->AsDouble();
    ASSERT_TRUE(cnt_d.has_value());
    EXPECT_EQ(static_cast<std::size_t>(*cnt_d), 5u);

    // Top-1 should be index 0 (closest to {0,0,0})
    auto idx1 = fact.Get("retrieval.top1.index");
    ASSERT_TRUE(idx1.has_value());
    EXPECT_DOUBLE_EQ(idx1->AsDouble().value(), 0.0);

    // FAISS IndexFlatL2 returns squared L2 distance.
    // For {0.1,0.1,0.1} vs {0,0,0}: squared L2 = 0.03, but float rounding
    // on 0.1F may produce a value slightly above 0.03. Use a relaxed bound.
    auto dist1 = fact.Get("retrieval.top1.distance");
    ASSERT_TRUE(dist1.has_value());
    EXPECT_NEAR(dist1->AsDouble().value(), 0.03, 0.005);

    // Top-2 should be index 1
    auto idx2 = fact.Get("retrieval.top2.index");
    ASSERT_TRUE(idx2.has_value());
    EXPECT_DOUBLE_EQ(idx2->AsDouble().value(), 1.0);

    // Sources should be VectorSearch kind
    auto& src = fact.SourceOf("retrieval.top1.index");
    EXPECT_EQ(src.kind, FactSourceKind::VectorSearch);
}

TEST(FactBuilder_RunVectorRetrieval, MissingKey) {
    FactBuilder fb(nullptr, nullptr);  // no VectorPath — should fail before Search

    FactBase fact;
    auto result = fb.RunVectorRetrieval(fact, "nonexistent_key");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_InvalidPath);
}

TEST(FactBuilder_RunVectorRetrieval, ConfigKOverride) {
    TestVectorPath tvp;
    ASSERT_TRUE(static_cast<bool>(tvp));

    auto vp_shared = std::make_shared<retrieval::VectorPath>(*tvp.vp);
    FactBuilder fb(nullptr, vp_shared);

    FactBase fact;
    fact.Set("embedding.test",
             Value::OfList({Value::Of(0.0), Value::Of(0.0), Value::Of(0.0)}),
             FactSource{});
    // Override k to 2
    fact.Set("retrieval.config.k", Value::Of(2.0), FactSource{});

    auto result = fb.RunVectorRetrieval(fact, "embedding.test");
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto count = fact.Get("retrieval.count");
    ASSERT_TRUE(count.has_value());
    EXPECT_DOUBLE_EQ(count->AsDouble().value(), 2.0);
}

// ── Integration test: Build with all sources ─────────────────────────

TEST(FactBuilder_Build, IntegrationWithGraphPath) {
    TestGraph tg;

    // Insert a root cause category for a defect type
    knowledge::KnowledgeRecord root_props;
    root_props.fields["name"] = std::string("RootCause-C");
    root_props.fields["weight"] = 0.75;
    auto root_id = tg.kg.InsertNode("RootCause", root_props);
    ASSERT_TRUE(root_id.has_value());

    knowledge::KnowledgeRecord def_props;
    def_props.fields["type_code"] = std::string("D-001");
    auto def_id = tg.kg.InsertNode("DefectType", def_props);
    ASSERT_TRUE(def_id.has_value());

    tg.kg.InsertEdge(*def_id, *root_id, "has_root_cause", {});

    auto kg_shared = std::make_shared<knowledge::KnowledgeGraph>(std::move(tg.kg));
    FactBuilder fb(kg_shared, nullptr);

    auto dr = MakeDetection(0.73F, 1);
    auto result = fb.Build("seat-42", dr, {"RootCause->weight"});
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // 1. Detection fields present
    EXPECT_TRUE(result->Has("detection.image_level_score"));
    EXPECT_TRUE(result->Has("detection.region_count"));
    EXPECT_TRUE(result->Has("detection.anomaly_map.max_score"));

    // 2. Graph path resolution
    auto weight = result->Get("graph.RootCause.weight");
    ASSERT_TRUE(weight.has_value());
    EXPECT_DOUBLE_EQ(weight->AsDouble().value(), 0.75);

    // 3. surface.id present
    auto sid = result->Get("surface.id");
    ASSERT_TRUE(sid.has_value());
    ASSERT_TRUE(sid->AsString().has_value());
    EXPECT_EQ(*sid->AsString(), "seat-42");

    // 4. All entries
    auto all = result->AllEntries();
    EXPECT_GE(all.size(), 10u);  // many fields from detection + graph + surface
}

}  // namespace
}  // namespace sai::rule
