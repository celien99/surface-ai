#include "parser.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace sai::rule {

namespace {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

std::string_view TokenTypeName(TokenType t) {
    using namespace std::string_view_literals;
    switch (t) {
    case TokenType::Number:     return "Number"sv;
    case TokenType::String:     return "String"sv;
    case TokenType::Bool:       return "Bool"sv;
    case TokenType::Identifier: return "Identifier"sv;
    case TokenType::Arrow:      return "'->'"sv;
    case TokenType::Dot:        return "'.'"sv;
    case TokenType::Plus:       return "'+'"sv;
    case TokenType::Minus:      return "'-'"sv;
    case TokenType::Star:       return "'*'"sv;
    case TokenType::Slash:      return "'/'"sv;
    case TokenType::Eq:         return "'=='"sv;
    case TokenType::Neq:        return "'!='"sv;
    case TokenType::Lt:         return "'<'"sv;
    case TokenType::Le:         return "'<='"sv;
    case TokenType::Gt:         return "'>'"sv;
    case TokenType::Ge:         return "'>='"sv;
    case TokenType::And:        return "AND"sv;
    case TokenType::Or:         return "OR"sv;
    case TokenType::Not:        return "NOT"sv;
    case TokenType::In:         return "IN"sv;
    case TokenType::LParen:     return "'('"sv;
    case TokenType::RParen:     return "')'"sv;
    case TokenType::LBracket:   return "'['"sv;
    case TokenType::RBracket:   return "']'"sv;
    case TokenType::Comma:      return "','"sv;
    case TokenType::End:        return "End"sv;
    }
    return "Unknown"sv;
}

auto MakeError(ErrorCode code, std::string msg) -> ErrorInfo {
    return ErrorInfo{code, std::move(msg), std::source_location::current()};
}

// Try to convert a string to a double, returns nullopt on failure.
std::optional<double> TryParseDouble(std::string_view s) {
    double val{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec == std::errc{} && ptr == s.data() + s.size()) return val;
    return std::nullopt;
}

}  // anonymous namespace

// ===========================================================================
// Parser
// ===========================================================================

Parser::Parser(std::vector<Token> tokens)
    : tokens_(std::move(tokens))
    , current_(0)
{
}

// ---------------------------------------------------------------------------
// Token stream helpers
// ---------------------------------------------------------------------------

auto Parser::Peek() const -> const Token& {
    return tokens_[current_];
}

auto Parser::Advance() -> Token {
    return tokens_[current_++];
}

auto Parser::Expect(TokenType t) -> Result<Token> {
    if (IsAtEnd()) {
        return tl::make_unexpected(MakeError(
            ErrorCode::Rule_ParseError,
            std::string("Expected ") + std::string(TokenTypeName(t))
                + " but reached end of expression"));
    }
    if (Peek().type != t) {
        return tl::make_unexpected(MakeError(
            ErrorCode::Rule_ParseError,
            std::string("Expected ") + std::string(TokenTypeName(t))
                + " but got '" + Peek().text + "'"));
    }
    return Advance();
}

bool Parser::IsAtEnd() const {
    return current_ >= tokens_.size();
}

// ===========================================================================
// Recursive descent parsing
// ===========================================================================

auto Parser::Parse() -> Result<std::unique_ptr<IExpression>> {
    return ParseOr();
}

// --- OR (lowest precedence) ---
auto Parser::ParseOr() -> Result<std::unique_ptr<IExpression>> {
    auto lhs = ParseAnd();
    if (!lhs.has_value()) return lhs;

    while (!IsAtEnd() && Peek().type == TokenType::Or) {
        Advance();
        auto rhs = ParseAnd();
        if (!rhs.has_value()) return rhs;
        lhs = std::make_unique<BinaryExpr>(
            BinaryOp::Or,
            std::move(*lhs),
            std::move(*rhs),
            "");
    }
    return lhs;
}

// --- AND ---
auto Parser::ParseAnd() -> Result<std::unique_ptr<IExpression>> {
    auto lhs = ParseNot();
    if (!lhs.has_value()) return lhs;

    while (!IsAtEnd() && Peek().type == TokenType::And) {
        Advance();
        auto rhs = ParseNot();
        if (!rhs.has_value()) return rhs;
        lhs = std::make_unique<BinaryExpr>(
            BinaryOp::And,
            std::move(*lhs),
            std::move(*rhs),
            "");
    }
    return lhs;
}

// --- NOT (unary prefix) ---
auto Parser::ParseNot() -> Result<std::unique_ptr<IExpression>> {
    if (!IsAtEnd() && Peek().type == TokenType::Not) {
        Advance();
        auto operand = ParseCompare();
        if (!operand.has_value()) return operand;
        return std::make_unique<UnaryExpr>(
            UnaryOp::Not,
            std::move(*operand),
            "");
    }
    return ParseCompare();
}

// --- Comparison: ==  !=  >  <  >=  <=  IN  ---
auto Parser::ParseCompare() -> Result<std::unique_ptr<IExpression>> {
    auto lhs = ParseAddSub();
    if (!lhs.has_value()) return lhs;

    if (IsAtEnd()) return lhs;

    // Determine the binary op from the current token (if it is a comparison)
    BinaryOp op{};
    bool is_compare = true;
    switch (Peek().type) {
    case TokenType::Eq:  op = BinaryOp::Eq;  break;
    case TokenType::Neq: op = BinaryOp::Neq; break;
    case TokenType::Lt:  op = BinaryOp::Lt;  break;
    case TokenType::Le:  op = BinaryOp::Le;  break;
    case TokenType::Gt:  op = BinaryOp::Gt;  break;
    case TokenType::Ge:  op = BinaryOp::Ge;  break;
    case TokenType::In:  op = BinaryOp::In;  break;
    default:
        is_compare = false;
        break;
    }

    if (is_compare) {
        Advance();
        auto rhs = ParseAddSub();
        if (!rhs.has_value()) return rhs;
        lhs = std::make_unique<BinaryExpr>(
            op,
            std::move(*lhs),
            std::move(*rhs),
            "");
    }
    return lhs;
}

// --- Addition / Subtraction:  +  -  (left-associative) ---
auto Parser::ParseAddSub() -> Result<std::unique_ptr<IExpression>> {
    auto lhs = ParseMulDiv();
    if (!lhs.has_value()) return lhs;

    while (!IsAtEnd()) {
        BinaryOp op{};
        bool is_addsub = true;
        switch (Peek().type) {
        case TokenType::Plus:  op = BinaryOp::Add; break;
        case TokenType::Minus: op = BinaryOp::Sub; break;
        default: is_addsub = false; break;
        }
        if (!is_addsub) break;

        Advance();
        auto rhs = ParseMulDiv();
        if (!rhs.has_value()) return rhs;
        lhs = std::make_unique<BinaryExpr>(
            op,
            std::move(*lhs),
            std::move(*rhs),
            "");
    }
    return lhs;
}

// --- Multiplication / Division:  *  /  (left-associative) ---
auto Parser::ParseMulDiv() -> Result<std::unique_ptr<IExpression>> {
    auto lhs = ParseUnary();
    if (!lhs.has_value()) return lhs;

    while (!IsAtEnd()) {
        BinaryOp op{};
        bool is_muldiv = true;
        switch (Peek().type) {
        case TokenType::Star:  op = BinaryOp::Mul; break;
        case TokenType::Slash: op = BinaryOp::Div; break;
        default: is_muldiv = false; break;
        }
        if (!is_muldiv) break;

        Advance();
        auto rhs = ParseUnary();
        if (!rhs.has_value()) return rhs;
        lhs = std::make_unique<BinaryExpr>(
            op,
            std::move(*lhs),
            std::move(*rhs),
            "");
    }
    return lhs;
}

// --- Unary:  + expr  |  - expr  ---
auto Parser::ParseUnary() -> Result<std::unique_ptr<IExpression>> {
    if (!IsAtEnd()) {
        if (Peek().type == TokenType::Minus) {
            Advance();
            auto operand = ParseUnary();
            if (!operand.has_value()) return operand;
            return std::make_unique<UnaryExpr>(
                UnaryOp::Neg,
                std::move(*operand),
                "");
        }
        if (Peek().type == TokenType::Plus) {
            Advance();  // unary plus is a no-op
            return ParseUnary();
        }
    }
    return ParseAtom();
}

// --- Atom (leaf nodes) ---
auto Parser::ParseAtom() -> Result<std::unique_ptr<IExpression>> {
    if (IsAtEnd()) {
        return tl::make_unexpected(MakeError(
            ErrorCode::Rule_ParseError,
            "Unexpected end of expression"));
    }

    const Token& tok = Peek();

    // --- Number literal ---
    if (tok.type == TokenType::Number) {
        Advance();
        auto d = TryParseDouble(tok.text);
        if (!d.has_value()) {
            return tl::make_unexpected(MakeError(
                ErrorCode::Rule_ParseError,
                "Invalid numeric literal: '" + tok.text + "'"));
        }
        return std::make_unique<LiteralExpr>(Value::Of(*d));
    }

    // --- String literal ---
    if (tok.type == TokenType::String) {
        Advance();
        return std::make_unique<LiteralExpr>(Value::Of(tok.text));
    }

    // --- Bool literal ---
    if (tok.type == TokenType::Bool) {
        Advance();
        return std::make_unique<LiteralExpr>(Value::Of(tok.text == "true"));
    }

    // --- Identifier → FieldRef, PathExpr, or FunctionCall ---
    if (tok.type == TokenType::Identifier) {
        Advance();  // consume the first identifier
        std::string name = tok.text;

        // Function call: identifier (
        if (!IsAtEnd() && Peek().type == TokenType::LParen) {
            return ParseFunctionCall(name);
        }

        // Collect -> and . chain:  a -> b -> c . d ...
        std::string path = std::move(name);
        bool has_arrow = false;

        while (!IsAtEnd()) {
            if (Peek().type == TokenType::Arrow) {
                has_arrow = true;
                Advance();  // consume ->
                if (IsAtEnd() || Peek().type != TokenType::Identifier) {
                    return tl::make_unexpected(MakeError(
                        ErrorCode::Rule_ParseError,
                        "Expected identifier after '->'"));
                }
                path += "->";
                path += Advance().text;
            } else if (Peek().type == TokenType::Dot) {
                Advance();  // consume .
                if (IsAtEnd() || Peek().type != TokenType::Identifier) {
                    return tl::make_unexpected(MakeError(
                        ErrorCode::Rule_ParseError,
                        "Expected identifier after '.'"));
                }
                path += ".";
                path += Advance().text;
            } else {
                break;
            }
        }

        if (has_arrow) {
            return std::make_unique<PathExpr>(std::move(path));
        }
        return std::make_unique<FieldRefExpr>(std::move(path));
    }

    // --- Parenthesized expression: ( expr ) ---
    if (tok.type == TokenType::LParen) {
        Advance();  // consume (
        auto expr = ParseOr();
        if (!expr.has_value()) return expr;
        auto rparen = Expect(TokenType::RParen);
        if (!rparen.has_value()) return tl::unexpected(rparen.error());
        return expr;
    }

    // --- List literal: [ val, val, ... ] ---
    if (tok.type == TokenType::LBracket) {
        return ParseListLiteral();
    }

    // --- Anything else is a syntax error ---
    return tl::make_unexpected(MakeError(
        ErrorCode::Rule_ParseError,
        std::string("Unexpected token '") + tok.text
            + "' (type: " + std::string(TokenTypeName(tok.type)) + ")"));
}

// --- Function call:  name ( arg, arg, ... ) ---
auto Parser::ParseFunctionCall(std::string name)
    -> Result<std::unique_ptr<IExpression>>
{
    // Map function name to BuiltinFn
    BuiltinFn fn{};
    if (name == "AVG")         fn = BuiltinFn::Avg;
    else if (name == "COUNT")  fn = BuiltinFn::Count;
    else if (name == "MAX")    fn = BuiltinFn::Max;
    else if (name == "MIN")    fn = BuiltinFn::Min;
    else if (name == "SUM")    fn = BuiltinFn::Sum;
    else if (name == "LEN")    fn = BuiltinFn::Len;
    else {
        return tl::make_unexpected(MakeError(
            ErrorCode::Rule_ParseError,
            "Unknown function: '" + name + "'"));
    }

    // LParen was NOT consumed yet (ParseAtom peeked but didn't advance)
    auto lparen = Expect(TokenType::LParen);
    if (!lparen.has_value()) return tl::unexpected(lparen.error());

    // Parse comma-separated argument expressions
    std::vector<std::unique_ptr<IExpression>> args;
    if (!IsAtEnd() && Peek().type != TokenType::RParen) {
        auto first = ParseOr();
        if (!first.has_value()) return tl::unexpected(first.error());
        args.push_back(std::move(*first));

        while (!IsAtEnd() && Peek().type == TokenType::Comma) {
            Advance();  // consume comma
            auto arg = ParseOr();
            if (!arg.has_value()) return tl::unexpected(arg.error());
            args.push_back(std::move(*arg));
        }
    }

    auto rparen = Expect(TokenType::RParen);
    if (!rparen.has_value()) return tl::unexpected(rparen.error());

    return std::make_unique<FunctionExpr>(fn, std::move(args), "");
}

// --- List literal: [ expr, expr, ... ] ---
auto Parser::ParseListLiteral() -> Result<std::unique_ptr<IExpression>> {
    auto lbracket = Expect(TokenType::LBracket);
    if (!lbracket.has_value()) return tl::unexpected(lbracket.error());

    std::vector<Value> items;
    if (!IsAtEnd() && Peek().type != TokenType::RBracket) {
        // Parse first item
        auto first_item = ParseOr();
        if (!first_item.has_value()) return tl::unexpected(first_item.error());

        // For list items we store the value directly.  We only handle
        // LiteralExpr items at parse time; non-literal expressions are
        // evaluated (and their values extracted) at evaluation time (Task 6).
        if ((*first_item)->GetKind() == ExprKind::Literal) {
            items.push_back(
                static_cast<const LiteralExpr&>(**first_item).GetValue());
        } else {
            return tl::make_unexpected(MakeError(
                ErrorCode::Rule_ParseError,
                "List items must be literal values at parse time"));
        }

        while (!IsAtEnd() && Peek().type == TokenType::Comma) {
            Advance();  // consume comma
            auto item = ParseOr();
            if (!item.has_value()) return tl::unexpected(item.error());
            if ((*item)->GetKind() == ExprKind::Literal) {
                items.push_back(
                    static_cast<const LiteralExpr&>(**item).GetValue());
            } else {
                return tl::make_unexpected(MakeError(
                    ErrorCode::Rule_ParseError,
                    "List items must be literal values at parse time"));
            }
        }
    }

    auto rbracket = Expect(TokenType::RBracket);
    if (!rbracket.has_value()) return tl::unexpected(rbracket.error());

    return std::make_unique<LiteralExpr>(Value::OfList(std::move(items)));
}

}  // namespace sai::rule
