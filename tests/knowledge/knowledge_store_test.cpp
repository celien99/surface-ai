#include <sai/knowledge/knowledge_store.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

TEST(KnowledgeStoreTest, CreateWithMemoryDb) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());
    EXPECT_NE((*store)->DbHandle(), nullptr);
    // schema_version is a separate table, nodes starts empty
    EXPECT_EQ((*store)->Graph().NodeCount(), 0);
}

TEST(KnowledgeStoreTest, InsertAndGetNode) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    props.fields["material_code"] = std::string("LEATHER-001");
    auto id = (*store)->InsertNode("Material", props, "importer");
    ASSERT_TRUE(id.has_value());

    auto node = (*store)->GetNode(*id);
    ASSERT_TRUE(node.has_value());
    EXPECT_EQ(node->type, "Material");
}

TEST(KnowledgeStoreTest, EvolutionEnabled) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = true;
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto id = (*store)->InsertNode("Material", props, "test");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    EXPECT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).operation, EvolutionOp::Insert);
}

TEST(KnowledgeStoreTest, EvolutionDisabled) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = false;
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto id = (*store)->InsertNode("Material", props, "test");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    EXPECT_EQ(history->size(), 0);
}

TEST(KnowledgeStoreTest, SnapshotCreateAndList) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    (void)(*store)->InsertNode("Material", props);

    auto snap_id = (*store)->CreateSnapshot("baseline");
    ASSERT_TRUE(snap_id.has_value());

    auto snapshots = (*store)->ListSnapshots();
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->size(), 1);
    EXPECT_EQ(snapshots->at(0).label, "baseline");
}

TEST(KnowledgeStoreTest, TraverseGraph) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto mat = (*store)->InsertNode("Material", props);
    auto sup = (*store)->InsertNode("Supplier", props);
    (void)(*store)->InsertEdge(*mat, *sup, "supplied_by", props);

    auto paths = (*store)->Traverse(*mat, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup);
}

TEST(KnowledgeStoreTest, RecursiveTraverseMultiLevel) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto a = (*store)->InsertNode("A", props);
    auto b = (*store)->InsertNode("B", props);
    auto c = (*store)->InsertNode("C", props);
    auto d = (*store)->InsertNode("D", props);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(c.has_value());
    ASSERT_TRUE(d.has_value());

    // A -> B -> C -> D via "next" edges
    (void)(*store)->InsertEdge(*a, *b, "next", props);
    (void)(*store)->InsertEdge(*b, *c, "next", props);
    (void)(*store)->InsertEdge(*c, *d, "next", props);

    // Traverse with default max_depth=3: should reach A->B, B->C, C->D
    auto paths = (*store)->Traverse(*a, "next");
    ASSERT_TRUE(paths.has_value());
    EXPECT_EQ(paths->size(), 3);

    // max_depth=1: only direct neighbor
    auto shallow = (*store)->Traverse(*a, "next", 1);
    ASSERT_TRUE(shallow.has_value());
    ASSERT_EQ(shallow->size(), 1);
    EXPECT_EQ(shallow->at(0).targets[0].id, *b);

    // max_depth=0: no traversal
    auto none = (*store)->Traverse(*a, "next", 0);
    ASSERT_TRUE(none.has_value());
    EXPECT_EQ(none->size(), 0);
}

TEST(KnowledgeStoreTest, LinkEmbedding) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto id = (*store)->InsertNode("Material", props);
    ASSERT_TRUE(id.has_value());

    auto result = (*store)->LinkEmbedding(*id, 42);
    EXPECT_TRUE(result.has_value());

    // Linking again (INSERT OR REPLACE) should also succeed
    auto result2 = (*store)->LinkEmbedding(*id, 99);
    EXPECT_TRUE(result2.has_value());
}

}  // namespace
}  // namespace sai::knowledge
