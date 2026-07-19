// condition_builder.cpp — Build IExpression AST directly from structured YAML.
// Batch T3 refactor: replaces Lexer+Parser text DSL with declarative YAML conditions.
//
// Supported YAML formats:
//   Simple comparison:  {field: "detection.score", op: "gt", value: 0.8}
//   AND junction:       {and: [{field: a, op: gt, value: 0.5}, ...]}
//   OR junction:        {or:  [{field: a, op: gt, value: 0.5}, ...]}
//   NOT negation:       {not: {field: a, op: eq, value: 0}}
//   Path expression:    {path: "material->supplier->batch"}  (returns PathExpr)
//   Literal value:      {value: "some_string"}  or  {value: 42}

#include "condition_builder.h"

#include <cctype>
#include <source_location>
#include <string>

#include <sai/rule/expression.h>
#include <sai/rule/ast_nodes.h>
#include <yaml-cpp/yaml.h>

namespace sai::rule {
namespace {

// Map operator string to BinaryOp
auto ParseBinaryOp(std::string_view op_str) -> std::optional<BinaryOp> {
    if (op_str == "and") return BinaryOp::And;
    if (op_str == "or")  return BinaryOp::Or;
    if (op_str == "add") return BinaryOp::Add;
    if (op_str == "sub") return BinaryOp::Sub;
    if (op_str == "mul") return BinaryOp::Mul;
    if (op_str == "div") return BinaryOp::Div;
    if (op_str == "eq")  return BinaryOp::Eq;
    if (op_str == "neq") return BinaryOp::Neq;
    if (op_str == "lt")  return BinaryOp::Lt;
    if (op_str == "le")  return BinaryOp::Le;
    if (op_str == "gt")  return BinaryOp::Gt;
    if (op_str == "ge")  return BinaryOp::Ge;
    if (op_str == "in")  return BinaryOp::In;
    return std::nullopt;
}

// Build a LiteralExpr from a YAML scalar value
auto BuildLiteral(const YAML::Node& val_node) -> Result<std::unique_ptr<IExpression>> {
    if (!val_node.IsScalar()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            "literal value must be a scalar",
            std::source_location::current()});
    }
    // Try int, then double, then bool, then string
    std::string text = val_node.as<std::string>();
    try {
        if (text == "true" || text == "false") {
            return std::make_unique<LiteralExpr>(Value::Of(text == "true"));
        }
        // Check if it's a number
        bool has_dot = (text.find('.') != std::string::npos);
        if (has_dot) {
            return std::make_unique<LiteralExpr>(Value::Of(val_node.as<double>()));
        }
        // Try int64
        if (std::all_of(text.begin(), text.end(), [](char c) {
                return std::isdigit(c) || c == '-'; })) {
            return std::make_unique<LiteralExpr>(
                Value::Of(val_node.as<std::int64_t>()));
        }
        // Fallback: string
        return std::make_unique<LiteralExpr>(Value::Of(text));
    } catch (const YAML::Exception& e) {
        return std::make_unique<LiteralExpr>(Value::Of(text));
    }
}

// Build a FieldRefExpr/PathExpr or comparison BinaryExpr from a field/path+op+value map
auto BuildComparison(const YAML::Node& node, std::string_view source_hint)
    -> Result<std::unique_ptr<IExpression>> {
    auto field_node = node["field"];
    auto path_node = node["path"];
    auto op_node = node["op"];
    auto value_node = node["value"];

    // Build the LHS expression: field → FieldRefExpr, path → PathExpr
    std::unique_ptr<IExpression> lhs;
    std::string lhs_source;

    if (field_node.IsScalar()) {
        lhs_source = field_node.as<std::string>();
        lhs = std::make_unique<FieldRefExpr>(lhs_source);
    } else if (path_node.IsScalar()) {
        lhs_source = path_node.as<std::string>();
        lhs = std::make_unique<PathExpr>(lhs_source);
    } else {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            std::string(source_hint) + ": must have 'field' or 'path' key",
            std::source_location::current()});
    }

    // If no op/value, treat as bare field/path reference
    if (!op_node.IsDefined() && !value_node.IsDefined()) {
        return lhs;
    }

    if (!op_node.IsScalar()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            std::string(source_hint) + ": 'op' must be a string",
            std::source_location::current()});
    }

    auto op = ParseBinaryOp(op_node.as<std::string>());
    if (!op.has_value()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            std::string(source_hint) + ": unknown operator '" +
                op_node.as<std::string>() + "'",
            std::source_location::current()});
    }

    auto value_result = BuildLiteral(value_node);
    if (!value_result.has_value()) {
        return tl::make_unexpected(value_result.error());
    }

    return std::make_unique<BinaryExpr>(
        *op, std::move(lhs), std::move(*value_result), lhs_source);
}

}  // anonymous namespace

// ── Public entry point ────────────────────────────────────────────────────

auto BuildExpressionFromYAML(const YAML::Node& node) -> Result<std::unique_ptr<IExpression>> {
    if (!node.IsDefined() || node.IsNull()) {
        return std::unique_ptr<IExpression>(nullptr);
    }

    // String → literal
    if (node.IsScalar()) {
        return BuildLiteral(node);
    }

    if (!node.IsMap()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            "condition must be a map or scalar",
            std::source_location::current()});
    }

    // {field: "...", op: "...", value: ...}  or  {path: "...", op: "...", value: ...}
    if (node["field"].IsDefined() || node["path"].IsDefined()) {
        return BuildComparison(node, "condition");
    }

    // {and: [expr1, expr2, ...]}
    if (node["and"].IsDefined()) {
        auto seq = node["and"];
        if (!seq.IsSequence() || seq.size() == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Rule_ParseError,
                "'and' must be a non-empty sequence",
                std::source_location::current()});
        }
        auto left = BuildExpressionFromYAML(seq[0]);
        if (!left.has_value()) return tl::make_unexpected(left.error());
        for (std::size_t i = 1; i < seq.size(); ++i) {
            auto right = BuildExpressionFromYAML(seq[i]);
            if (!right.has_value()) return tl::make_unexpected(right.error());
            left = std::make_unique<BinaryExpr>(
                BinaryOp::And, std::move(*left), std::move(*right), "and");
        }
        return left;
    }

    // {or: [expr1, expr2, ...]}
    if (node["or"].IsDefined()) {
        auto seq = node["or"];
        if (!seq.IsSequence() || seq.size() == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Rule_ParseError,
                "'or' must be a non-empty sequence",
                std::source_location::current()});
        }
        auto left = BuildExpressionFromYAML(seq[0]);
        if (!left.has_value()) return tl::make_unexpected(left.error());
        for (std::size_t i = 1; i < seq.size(); ++i) {
            auto right = BuildExpressionFromYAML(seq[i]);
            if (!right.has_value()) return tl::make_unexpected(right.error());
            left = std::make_unique<BinaryExpr>(
                BinaryOp::Or, std::move(*left), std::move(*right), "or");
        }
        return left;
    }

    // {not: expr}
    if (node["not"].IsDefined()) {
        auto operand = BuildExpressionFromYAML(node["not"]);
        if (!operand.has_value()) return tl::make_unexpected(operand.error());
        return std::make_unique<UnaryExpr>(
            UnaryOp::Not, std::move(*operand), "not");
    }

    // {value: ...} → literal (bare value)
    if (node["value"].IsDefined()) {
        return BuildLiteral(node["value"]);
    }

    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Rule_ParseError,
        "unrecognized condition structure — expected 'field'+'op'+'value', "
        "'and'/'or'/'not', 'path', or 'value'",
        std::source_location::current()});
}

}  // namespace sai::rule
