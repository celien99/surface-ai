#include "../src/rule/parser.h"
#include "../src/rule/lexer.h"
#include <sai/rule/ast_nodes.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

// ---------------------------------------------------------------------------
// Helper: parse source string into an expression.  Convenience for tests
// that expect success.
// ---------------------------------------------------------------------------
auto ParseExpr(std::string_view source) -> std::unique_ptr<IExpression> {
    Lexer lex(source);
    auto tokens = lex.Tokenize();
    EXPECT_TRUE(tokens.has_value());
    Parser parser(std::move(*tokens));
    auto expr = parser.Parse();
    EXPECT_TRUE(expr.has_value());
    return std::move(*expr);
}

// ===========================================================================
// Literals
// ===========================================================================

TEST(ParserTest, ParseLiteralDouble) {
    auto expr = ParseExpr("3.14");
    EXPECT_EQ(expr->GetKind(), ExprKind::Literal);
    EXPECT_EQ(static_cast<LiteralExpr*>(expr.get())->GetValue().AsDouble().value(), 3.14);
}

TEST(ParserTest, ParseLiteralString) {
    auto expr = ParseExpr(R"("hello")");
    EXPECT_EQ(expr->GetKind(), ExprKind::Literal);
    EXPECT_EQ(static_cast<LiteralExpr*>(expr.get())->GetValue().AsString().value(), "hello");
}

TEST(ParserTest, ParseLiteralBool) {
    auto expr = ParseExpr("true");
    EXPECT_EQ(expr->GetKind(), ExprKind::Literal);
    EXPECT_TRUE(static_cast<LiteralExpr*>(expr.get())->GetValue().AsBool().value());
}

// ===========================================================================
// Field reference & path expression
// ===========================================================================

TEST(ParserTest, ParseFieldRef) {
    auto expr = ParseExpr("defect.score");
    EXPECT_EQ(expr->GetKind(), ExprKind::FieldRef);
    auto refs = expr->CollectFieldRefs();
    ASSERT_EQ(refs.size(), 1);
    EXPECT_EQ(refs[0], "defect.score");
}

TEST(ParserTest, ParsePathExpr) {
    auto expr = ParseExpr("material->supplier->batch.reject_rate");
    EXPECT_EQ(expr->GetKind(), ExprKind::Path);
}

// ===========================================================================
// Arithmetic
// ===========================================================================

TEST(ParserTest, ParseArithmetic) {
    auto expr = ParseExpr("defect.score * 2.0 + 1.0");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Add);
}

// ===========================================================================
// Comparison
// ===========================================================================

TEST(ParserTest, ParseComparison) {
    auto expr = ParseExpr("defect.score > 0.8");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Gt);
}

// ===========================================================================
// AND / OR
// ===========================================================================

TEST(ParserTest, ParseAndOr) {
    auto expr = ParseExpr("defect.score > 0.8 AND defect.area_mm2 > 10");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::And);
}

// ===========================================================================
// NOT
// ===========================================================================

TEST(ParserTest, ParseNot) {
    auto expr = ParseExpr("NOT defect.passed");
    EXPECT_EQ(expr->GetKind(), ExprKind::Unary);
}

// ===========================================================================
// IN (list membership)
// ===========================================================================

TEST(ParserTest, ParseInList) {
    auto expr = ParseExpr(R"(defect.position IN ["center", "left"])");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::In);
}

// ===========================================================================
// Function call
// ===========================================================================

TEST(ParserTest, ParseFunctionCall) {
    auto expr = ParseExpr("AVG(1.0, 2.0, 3.0)");
    EXPECT_EQ(expr->GetKind(), ExprKind::Function);
}

// ===========================================================================
// Parenthesized expressions
// ===========================================================================

TEST(ParserTest, ParseParenthesized) {
    auto expr = ParseExpr("(defect.score + 1.0) * 2.0");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Mul);
}

// ===========================================================================
// Operator precedence: AND binds tighter than OR
// ===========================================================================

TEST(ParserTest, ParsePrecedence) {
    // a OR b AND c  →  a OR (b AND c)
    auto expr = ParseExpr("a OR b AND c");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Or);

    const auto& lhs = static_cast<const BinaryExpr&>(*expr).GetLhs();
    EXPECT_EQ(lhs.GetKind(), ExprKind::FieldRef);

    const auto& rhs = static_cast<const BinaryExpr&>(*expr).GetRhs();
    EXPECT_EQ(rhs.GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<const BinaryExpr&>(rhs).GetOp(), BinaryOp::And);
}

// ===========================================================================
// Complex rule condition (multiple ANDs left-associative)
// ===========================================================================

TEST(ParserTest, ComplexRuleCondition) {
    auto expr = ParseExpr(
        R"(defect.type == "scratch" AND defect.area_mm2 > 10 AND defect.position IN ["center", "left"])");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    // Should be ((type=="scratch" AND area>10) AND position IN [...])
}

// ===========================================================================
// Syntax error returns Rule_ParseError
// ===========================================================================

TEST(ParserTest, SyntaxErrorReturnsParseError) {
    Lexer lex("defect >");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    Parser parser(std::move(*tokens));
    auto expr = parser.Parse();
    EXPECT_FALSE(expr.has_value());
    EXPECT_EQ(expr.error().code, ErrorCode::Rule_ParseError);
}

}  // namespace
}  // namespace sai::rule
