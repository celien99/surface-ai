#include <sai/knowledge/knowledge_graph.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

class KnowledgeGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        graph_ = KnowledgeGraph(db_);
        // create tables manually for unit test (KnowledgeStore does this in production)
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')));
            CREATE TABLE edges (id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, relationship TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT (datetime('now')));
            CREATE INDEX idx_nodes_type ON nodes(type);
            CREATE INDEX idx_edges_source ON edges(source_id);
            CREATE INDEX idx_edges_target ON edges(target_id);
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
    }
    void TearDown() override {
        sqlite3_close(db_);
    }
    sqlite3* db_ = nullptr;
    KnowledgeGraph graph_{nullptr};
};

TEST_F(KnowledgeGraphTest, InsertAndGetNode) {
    KnowledgeRecord props;
    props.fields["name"] = std::string("Nappa Leather");
    props.fields["code"] = std::string("LEATHER-001");

    auto id = graph_.InsertNode("Material", props);
    ASSERT_TRUE(id.has_value());

    auto node = graph_.GetNode(*id);
    ASSERT_TRUE(node.has_value());
    EXPECT_EQ(node->type, "Material");
    EXPECT_EQ(std::get<std::string>(node->properties.fields["name"]), "Nappa Leather");
}

TEST_F(KnowledgeGraphTest, NodeNotFound) {
    auto node = graph_.GetNode(99999);
    EXPECT_FALSE(node.has_value());
    EXPECT_EQ(node.error().code, ErrorCode::Knowledge_NodeNotFound);
}

TEST_F(KnowledgeGraphTest, FindNodesByType) {
    KnowledgeRecord props;
    graph_.InsertNode("Material", props);
    graph_.InsertNode("Material", props);
    graph_.InsertNode("Supplier", props);

    auto materials = graph_.FindNodesByType("Material");
    ASSERT_TRUE(materials.has_value());
    EXPECT_EQ(materials->size(), 2);
}

TEST_F(KnowledgeGraphTest, InsertAndGetEdge) {
    KnowledgeRecord props;
    auto src = graph_.InsertNode("Material", props);
    auto dst = graph_.InsertNode("Supplier", props);

    KnowledgeRecord edge_props;
    edge_props.fields["since"] = std::int64_t(2026);
    auto edge = graph_.InsertEdge(*src, *dst, "supplied_by", edge_props);
    ASSERT_TRUE(edge.has_value());

    auto got = graph_.GetEdge(*edge);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->source_id, *src);
    EXPECT_EQ(got->target_id, *dst);
    EXPECT_EQ(got->relationship, "supplied_by");
}

TEST_F(KnowledgeGraphTest, Traverse) {
    KnowledgeRecord props;
    auto mat = graph_.InsertNode("Material", props);
    auto sup = graph_.InsertNode("Supplier", props);

    ASSERT_TRUE(mat.has_value());
    ASSERT_TRUE(sup.has_value());

    graph_.InsertEdge(*mat, *sup, "supplied_by", props);

    auto paths = graph_.Traverse(*mat, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->size(), 1);
    EXPECT_EQ(paths->at(0).source, *mat);
    EXPECT_EQ(paths->at(0).targets.size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup);
}

TEST_F(KnowledgeGraphTest, DeleteNode) {
    KnowledgeRecord props;
    auto id = graph_.InsertNode("Material", props);
    ASSERT_TRUE(id.has_value());

    auto result = graph_.DeleteNode(*id);
    EXPECT_TRUE(result.has_value());

    auto node = graph_.GetNode(*id);
    EXPECT_FALSE(node.has_value());
}

TEST_F(KnowledgeGraphTest, InsertEdgeInvalidRelationship) {
    KnowledgeRecord props;
    auto src = graph_.InsertNode("Material", props);
    auto dst = graph_.InsertNode("Supplier", props);

    auto edge = graph_.InsertEdge(*src, *dst, "", props);
    EXPECT_FALSE(edge.has_value());
    EXPECT_EQ(edge.error().code, ErrorCode::Knowledge_InvalidRelationship);
}

}  // anonymous namespace
}  // namespace sai::knowledge
