#include <sai/knowledge/knowledge_evolution.h>
#include <sqlite3.h>
#include <gtest/gtest.h>
#include <chrono>

namespace sai::knowledge {
namespace {

class KnowledgeEvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE evolution_log (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                version INTEGER NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                changed_by TEXT NOT NULL DEFAULT 'system',
                before_image_json TEXT
            );
            CREATE INDEX idx_evolution_entity ON evolution_log(entity_type, entity_id);
            CREATE INDEX idx_evolution_timestamp ON evolution_log(timestamp);
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        evolution_ = KnowledgeEvolution(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    KnowledgeEvolution evolution_{nullptr};
};

TEST_F(KnowledgeEvolutionTest, AppendAndGetHistory) {
    KnowledgeRecord before;
    before.fields["name"] = std::string("old_value");
    auto r = evolution_.Append("Node", 1, EvolutionOp::Update, before, "importer");
    ASSERT_TRUE(r.has_value());

    auto history = evolution_.GetHistory("Node", 1);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).entity_type, "Node");
    EXPECT_EQ(history->at(0).entity_id, 1);
    EXPECT_EQ(history->at(0).operation, EvolutionOp::Update);
    EXPECT_EQ(history->at(0).version, 1);
}

TEST_F(KnowledgeEvolutionTest, VersionIncrements) {
    KnowledgeRecord empty;
    evolution_.Append("Node", 1, EvolutionOp::Insert, empty, "test");
    evolution_.Append("Node", 1, EvolutionOp::Update, empty, "test");
    evolution_.Append("Node", 1, EvolutionOp::Delete, empty, "test");

    auto history = evolution_.GetHistory("Node", 1);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 3);
    EXPECT_EQ(history->at(0).version, 1);
    EXPECT_EQ(history->at(1).version, 2);
    EXPECT_EQ(history->at(2).version, 3);
}

TEST_F(KnowledgeEvolutionTest, GetChangesSince) {
    KnowledgeRecord empty;
    evolution_.Append("Node", 1, EvolutionOp::Insert, empty, "test");
    auto after_first = std::chrono::system_clock::now();
    evolution_.Append("Node", 2, EvolutionOp::Insert, empty, "test");

    auto changes = evolution_.GetChangesSince(after_first);
    ASSERT_TRUE(changes.has_value());
    ASSERT_GE(changes->size(), 1);
    EXPECT_EQ(changes->at(0).entity_id, 2);
}

}  // namespace
}  // namespace sai::knowledge
