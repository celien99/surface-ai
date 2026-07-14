#include <sai/rule/ast_nodes.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(AstNodeTest, LiteralExprStoresValue) {
    LiteralExpr lit(Value::Of(42.0));
    EXPECT_EQ(lit.GetKind(), ExprKind::Literal);
    EXPECT_EQ(lit.GetValue().AsDouble().value(), 42.0);
}

TEST(AstNodeTest, FieldRefExprStoresKey) {
    FieldRefExpr ref("defect.score");
    EXPECT_EQ(ref.GetKind(), ExprKind::FieldRef);
    auto refs = ref.CollectFieldRefs();
    ASSERT_EQ(refs.size(), 1);
    EXPECT_EQ(refs[0], "defect.score");
}

TEST(AstNodeTest, BinaryExprStoresOpAndChildren) {
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(2.0));
    BinaryExpr expr(BinaryOp::Add, std::move(lhs), std::move(rhs), "3 + 2");
    EXPECT_EQ(expr.GetKind(), ExprKind::Binary);
    EXPECT_EQ(expr.GetOp(), BinaryOp::Add);
}

TEST(AstNodeTest, PathExprFlattensArrowToDot) {
    PathExpr path("material->supplier->batch.reject_rate");
    EXPECT_EQ(path.GetKind(), ExprKind::Path);
    EXPECT_EQ(path.GetFlattenedKey(), "material.supplier.batch.reject_rate");
}

TEST(AstNodeTest, CollectFieldRefsFromBinaryExpr) {
    auto lhs = std::make_unique<FieldRefExpr>("defect.score");
    auto rhs = std::make_unique<FieldRefExpr>("defect.confidence");
    BinaryExpr expr(BinaryOp::And, std::move(lhs), std::move(rhs), "defect.score AND defect.confidence");
    auto refs = expr.CollectFieldRefs();
    EXPECT_EQ(refs.size(), 2);
}

}  // namespace
}  // namespace sai::rule
