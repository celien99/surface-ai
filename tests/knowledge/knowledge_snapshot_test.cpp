// knowledge_snapshot_test.cpp — SAVEPOINT 时间点快照单元测试
#include <sai/knowledge/knowledge_snapshot.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

class KnowledgeSnapshotTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                savepoint_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                node_count INTEGER NOT NULL DEFAULT 0,
                edge_count INTEGER NOT NULL DEFAULT 0
            );
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
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        snapshot_ = KnowledgeSnapshot(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    KnowledgeSnapshot snapshot_{nullptr};
};

TEST_F(KnowledgeSnapshotTest, CreateAndList) {
    auto id = snapshot_.Create("baseline");
    ASSERT_TRUE(id.has_value());

    auto snapshots = snapshot_.List();
    ASSERT_TRUE(snapshots.has_value());
    ASSERT_EQ(snapshots->size(), 1);
    EXPECT_EQ(snapshots->at(0).label, "baseline");
}

TEST_F(KnowledgeSnapshotTest, RestoreRevertsChanges) {
    // Insert a node before snapshot
    sqlite3_exec(db_, "INSERT INTO nodes (type, properties_json) VALUES ('Test', '{}')", nullptr, nullptr, nullptr);

    auto id = snapshot_.Create("before_delete");
    ASSERT_TRUE(id.has_value());

    // Delete the node after snapshot
    sqlite3_exec(db_, "DELETE FROM nodes WHERE id = 1", nullptr, nullptr, nullptr);

    // Verify deleted
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 0);
    sqlite3_finalize(stmt);

    // Restore snapshot
    auto result = snapshot_.Restore(*id);
    ASSERT_TRUE(result.has_value());

    // Verify node is back
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 1);
    sqlite3_finalize(stmt);
}

TEST_F(KnowledgeSnapshotTest, DeleteSnapshot) {
    auto id = snapshot_.Create("temp");
    ASSERT_TRUE(id.has_value());

    auto result = snapshot_.Delete(*id);
    EXPECT_TRUE(result.has_value());

    auto snapshots = snapshot_.List();
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->size(), 0);
}

}  // namespace
}  // namespace sai::knowledge
