#include <sai/rule/ast_nodes.h>
#include <sai/rule/fact_base.h>
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

// ===========================================================================
// ExpressionEvalTest — exercises Evaluate() on all 6 AST node types
// ===========================================================================

TEST(ExpressionEvalTest, LiteralEvaluatesToItself) {
    FactBase fb;
    LiteralExpr lit(Value::Of(42.0));
    auto r = lit.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 42.0);
}

TEST(ExpressionEvalTest, FieldRefReadsFromFactBase) {
    FactBase fb;
    fb.Set("defect.score", Value::Of(0.85), FactSource{});
    FieldRefExpr ref("defect.score");
    auto r = ref.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 0.85);
}

TEST(ExpressionEvalTest, FieldRefMissingKeyReturnsError) {
    FactBase fb;
    FieldRefExpr ref("nonexistent");
    auto r = ref.Evaluate(fb);
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, ErrorCode::Rule_TypeMismatch);
}

TEST(ExpressionEvalTest, ArithmeticAdd) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(2.0));
    BinaryExpr expr(BinaryOp::Add, std::move(lhs), std::move(rhs), "3 + 2");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 5.0);
}

TEST(ExpressionEvalTest, ArithmeticSub) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(10.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    BinaryExpr expr(BinaryOp::Sub, std::move(lhs), std::move(rhs), "10 - 3");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 7.0);
}

TEST(ExpressionEvalTest, ArithmeticMul) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(4.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    BinaryExpr expr(BinaryOp::Mul, std::move(lhs), std::move(rhs), "4 * 5");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 20.0);
}

TEST(ExpressionEvalTest, ArithmeticDiv) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(7.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(2.0));
    BinaryExpr expr(BinaryOp::Div, std::move(lhs), std::move(rhs), "7 / 2");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 3.5);
}

TEST(ExpressionEvalTest, DivByZeroReturnsNull) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(0.0));
    BinaryExpr expr(BinaryOp::Div, std::move(lhs), std::move(rhs), "5 / 0");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->IsNull());
}

TEST(ExpressionEvalTest, ArithmeticOnNonNumericReturnsNull) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(std::string("hello")));
    BinaryExpr expr(BinaryOp::Add, std::move(lhs), std::move(rhs), "3 + hello");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->IsNull());
}

TEST(ExpressionEvalTest, ComparisonEq) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    BinaryExpr expr(BinaryOp::Eq, std::move(lhs), std::move(rhs), "5 == 5");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, ComparisonNeq) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    BinaryExpr expr(BinaryOp::Neq, std::move(lhs), std::move(rhs), "5 != 3");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, ComparisonGt) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    BinaryExpr expr(BinaryOp::Gt, std::move(lhs), std::move(rhs), "5 > 3");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, ComparisonLt) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(2.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(8.0));
    BinaryExpr expr(BinaryOp::Lt, std::move(lhs), std::move(rhs), "2 < 8");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, ComparisonLe) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(4.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(4.0));
    BinaryExpr expr(BinaryOp::Le, std::move(lhs), std::move(rhs), "4 <= 4");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, ComparisonGe) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    BinaryExpr expr(BinaryOp::Ge, std::move(lhs), std::move(rhs), "3 >= 3");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, AndShortCircuits) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(false));
    // rhs references missing key — should not be evaluated
    auto rhs = std::make_unique<FieldRefExpr>("nonexistent");
    BinaryExpr expr(BinaryOp::And, std::move(lhs), std::move(rhs),
                    "false AND nonexistent");
    auto r = expr.Evaluate(fb);
    // Short-circuit means rhs is never evaluated, so missing key doesn't error
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, OrShortCircuits) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(true));
    auto rhs = std::make_unique<FieldRefExpr>("nonexistent");
    BinaryExpr expr(BinaryOp::Or, std::move(lhs), std::move(rhs),
                    "true OR nonexistent");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, AndBothEvaluatedWhenLhsTrue) {
    FactBase fb;
    fb.Set("a", Value::Of(true), FactSource{});
    fb.Set("b", Value::Of(false), FactSource{});
    auto lhs = std::make_unique<FieldRefExpr>("a");
    auto rhs = std::make_unique<FieldRefExpr>("b");
    BinaryExpr expr(BinaryOp::And, std::move(lhs), std::move(rhs),
                    "a AND b");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, OrBothEvaluatedWhenLhsFalse) {
    FactBase fb;
    fb.Set("a", Value::Of(false), FactSource{});
    fb.Set("b", Value::Of(true), FactSource{});
    auto lhs = std::make_unique<FieldRefExpr>("a");
    auto rhs = std::make_unique<FieldRefExpr>("b");
    BinaryExpr expr(BinaryOp::Or, std::move(lhs), std::move(rhs),
                    "a OR b");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, NotOperator) {
    FactBase fb;
    auto operand = std::make_unique<LiteralExpr>(Value::Of(true));
    UnaryExpr expr(UnaryOp::Not, std::move(operand), "NOT true");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, NegOperator) {
    FactBase fb;
    auto operand = std::make_unique<LiteralExpr>(Value::Of(7.0));
    UnaryExpr expr(UnaryOp::Neg, std::move(operand), "-7");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), -7.0);
}

TEST(ExpressionEvalTest, InOperator) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(std::string("center")));
    auto rhs = std::make_unique<LiteralExpr>(
        Value::OfList({Value::Of(std::string("center")),
                       Value::Of(std::string("left"))}));
    BinaryExpr expr(BinaryOp::In, std::move(lhs), std::move(rhs),
                    R"(center IN [center, left])");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, InOperatorNotInList) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(std::string("right")));
    auto rhs = std::make_unique<LiteralExpr>(
        Value::OfList({Value::Of(std::string("center")),
                       Value::Of(std::string("left"))}));
    BinaryExpr expr(BinaryOp::In, std::move(lhs), std::move(rhs),
                    R"(right IN [center, left])");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, InWithNonListRhsReturnsFalse) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(1.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(2.0));
    BinaryExpr expr(BinaryOp::In, std::move(lhs), std::move(rhs),
                    "1 IN 2");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, FunctionAvg) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(1.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(2.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(3.0)));
    FunctionExpr expr(BuiltinFn::Avg, std::move(args), "AVG(1,2,3)");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 2.0);
}

TEST(ExpressionEvalTest, FunctionCount) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(10.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(20.0)));
    FunctionExpr expr(BuiltinFn::Count, std::move(args), "COUNT(10,20)");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 2.0);
}

TEST(ExpressionEvalTest, FunctionMax) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(1.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(5.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(3.0)));
    FunctionExpr expr(BuiltinFn::Max, std::move(args), "MAX(1,5,3)");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 5.0);
}

TEST(ExpressionEvalTest, FunctionMin) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(7.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(2.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(9.0)));
    FunctionExpr expr(BuiltinFn::Min, std::move(args), "MIN(7,2,9)");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 2.0);
}

TEST(ExpressionEvalTest, FunctionSum) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(4.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(5.0)));
    args.push_back(std::make_unique<LiteralExpr>(Value::Of(6.0)));
    FunctionExpr expr(BuiltinFn::Sum, std::move(args), "SUM(4,5,6)");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 15.0);
}

TEST(ExpressionEvalTest, FunctionLenOnString) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(
        std::make_unique<LiteralExpr>(Value::Of(std::string("hello"))));
    FunctionExpr expr(BuiltinFn::Len, std::move(args), R"(LEN("hello"))");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 5.0);
}

TEST(ExpressionEvalTest, FunctionLenOnList) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;
    args.push_back(std::make_unique<LiteralExpr>(
        Value::OfList({Value::Of(1.0), Value::Of(2.0), Value::Of(3.0)})));
    FunctionExpr expr(BuiltinFn::Len, std::move(args), "LEN([1,2,3])");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 3.0);
}

TEST(ExpressionEvalTest, FunctionLenEmptyReturnsZero) {
    FactBase fb;
    std::vector<std::unique_ptr<IExpression>> args;  // no args
    FunctionExpr expr(BuiltinFn::Len, std::move(args), "LEN()");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 0.0);
}

TEST(ExpressionEvalTest, PathExprReadsFlattenedKey) {
    FactBase fb;
    fb.AddPathMapping("material->supplier->batch.reject_rate",
                      "material.supplier.batch.reject_rate");
    fb.Set("material.supplier.batch.reject_rate", Value::Of(0.032),
           FactSource{});
    PathExpr path("material->supplier->batch.reject_rate");
    auto r = path.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 0.032);
}

TEST(ExpressionEvalTest, PathExprFallsBackToBuiltinFlatten) {
    FactBase fb;
    // No explicit mapping registered — PathExpr uses its own arrow→dot logic
    fb.Set("a.b.c", Value::Of(99.0), FactSource{});
    PathExpr path("a->b->c");
    auto r = path.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 99.0);
}

TEST(ExpressionEvalTest, PathExprMissingKeyReturnsError) {
    FactBase fb;
    PathExpr path("nothing.here");
    auto r = path.Evaluate(fb);
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, ErrorCode::Rule_TypeMismatch);
}

TEST(ExpressionEvalTest, MemoizationCachesResult) {
    FactBase fb;
    fb.Set("defect.score", Value::Of(0.5), FactSource{});
    FieldRefExpr ref("defect.score");
    auto r1 = ref.Evaluate(fb);
    ASSERT_TRUE(r1.has_value());
    EXPECT_DOUBLE_EQ(r1->AsDouble().value(), 0.5);

    // Verify a memoization key was stored in the FactBase
    bool memo_found = false;
    for (const auto& [k, v] : fb.AllEntries()) {
        if (k.rfind("__memo__", 0) == 0) {
            memo_found = true;
            break;
        }
    }
    EXPECT_TRUE(memo_found);
}

TEST(ExpressionEvalTest, MemoizationReturnsCachedValue) {
    FactBase fb;
    fb.Set("score", Value::Of(0.9), FactSource{});
    FieldRefExpr ref("score");

    // First evaluation computes and caches
    auto r1 = ref.Evaluate(fb);
    ASSERT_TRUE(r1.has_value());
    EXPECT_DOUBLE_EQ(r1->AsDouble().value(), 0.9);

    // Change the underlying FactBase value
    fb.Set("score", Value::Of(0.1), FactSource{});

    // Second evaluation should return the cached result (still 0.9)
    auto r2 = ref.Evaluate(fb);
    ASSERT_TRUE(r2.has_value());
    EXPECT_DOUBLE_EQ(r2->AsDouble().value(), 0.9);
}

}  // namespace
}  // namespace sai::rule
