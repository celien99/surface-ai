#include <sai/rule/fact_base.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(FactBaseTest, SetAndGet) {
    FactBase fb;
    fb.Set("defect.score", Value::Of(0.92), FactSource{FactSourceKind::Direct, "from detection"});
    auto v = fb.Get("defect.score");
    ASSERT_TRUE(v.has_value());
    EXPECT_DOUBLE_EQ(v->AsDouble().value(), 0.92);
}

TEST(FactBaseTest, GetMissingKeyReturnsNullopt) {
    FactBase fb;
    EXPECT_FALSE(fb.Get("nonexistent").has_value());
}

TEST(FactBaseTest, GetOrReturnsDefault) {
    FactBase fb;
    auto v = fb.GetOr("missing", Value::Of(0.5));
    EXPECT_DOUBLE_EQ(v.AsDouble().value(), 0.5);
}

TEST(FactBaseTest, HasKey) {
    FactBase fb;
    fb.Set("key", Value::Of(true), FactSource{});
    EXPECT_TRUE(fb.Has("key"));
    EXPECT_FALSE(fb.Has("other"));
}

TEST(FactBaseTest, SourceTracking) {
    FactBase fb;
    FactSource src{FactSourceKind::GraphPath, "Material→Supplier→Batch", std::chrono::microseconds{1200}, "SELECT ..."};
    fb.Set("material.supplier.name", Value::Of(std::string("SupplierA")), src);
    auto& tracked = fb.SourceOf("material.supplier.name");
    EXPECT_EQ(tracked.kind, FactSourceKind::GraphPath);
    EXPECT_EQ(tracked.sql.value(), "SELECT ...");
}

TEST(FactBaseTest, PathMapping) {
    FactBase fb;
    fb.AddPathMapping("material->supplier->batch.reject_rate", "material.supplier.batch.reject_rate");
    auto flat = fb.ResolvePath("material->supplier->batch.reject_rate");
    ASSERT_TRUE(flat.has_value());
    EXPECT_EQ(*flat, "material.supplier.batch.reject_rate");
}

TEST(FactBaseTest, SnapshotIsIndependent) {
    FactBase fb;
    fb.Set("a", Value::Of(1.0), FactSource{});
    auto snap = fb.Snapshot();
    fb.Set("b", Value::Of(2.0), FactSource{});
    EXPECT_TRUE(fb.Has("b"));
    EXPECT_FALSE(snap.Has("b"));  // snapshot doesn't see later writes
}

TEST(FactBaseTest, AllEntries) {
    FactBase fb;
    fb.Set("x", Value::Of(1.0), FactSource{});
    fb.Set("y", Value::Of(2.0), FactSource{});
    auto entries = fb.AllEntries();
    EXPECT_EQ(entries.size(), 2);
}

TEST(FactBaseTest, SetDefaultOnlyIfMissing) {
    FactBase fb;
    fb.SetDefault("a", Value::Of(10.0));
    fb.Set("a", Value::Of(20.0), FactSource{});
    fb.SetDefault("a", Value::Of(30.0));  // should NOT overwrite
    EXPECT_DOUBLE_EQ(fb.Get("a")->AsDouble().value(), 20.0);
}

}  // namespace
}  // namespace sai::rule
