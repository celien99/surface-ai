#include <sai/retrieval/metadata_path.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::retrieval {
namespace {

class MetadataPathTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"material_code":"LEATHER-001","grade":"A"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"material_code":"PVC-002","grade":"B"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Supplier', '{"supplier_code":"SUP-2026","name":"SupplierA"}');
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        meta_path_ = std::make_unique<MetadataPath>(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    std::unique_ptr<MetadataPath> meta_path_;
};

TEST_F(MetadataPathTest, FilterByType) {
    MetadataPath::Config cfg;
    cfg.node_types = {"Material"};

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), 2);
}

TEST_F(MetadataPathTest, FilterByFieldEqual) {
    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "material_code", FilterOp::Equal, std::string("LEATHER-001")
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 1);
    EXPECT_GT(results->at(0).score, 0.5F);
}

TEST_F(MetadataPathTest, FilterByIntGreaterThan) {
    // Insert nodes with integer fields
    sqlite3_exec(db_,
        "INSERT INTO nodes (type, properties_json) VALUES ('Batch', '{\"defect_count\":5}');"
        "INSERT INTO nodes (type, properties_json) VALUES ('Batch', '{\"defect_count\":15}');",
        nullptr, nullptr, nullptr);

    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "defect_count", FilterOp::GreaterThan, std::int64_t(10)
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 1);
}

TEST_F(MetadataPathTest, NoMatchReturnsEmpty) {
    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "nonexistent", FilterOp::Equal, std::string("NOPE")
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), 0);
}

}  // namespace
}  // namespace sai::retrieval
