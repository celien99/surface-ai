#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "sai/core/error.h"
#include "sai/rule/expression.h"
#include "sai/rule/ast_nodes.h"
#include "lexer.h"

namespace sai::rule {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);
    auto Parse() -> Result<std::unique_ptr<IExpression>>;

private:
    // Recursive descent — leftmost-token consumption order:
    // ParseOr -> ParseAnd -> ParseNot -> ParseCompare -> ParseAddSub
    //        -> ParseMulDiv -> ParseUnary -> ParseAtom
    auto ParseOr() -> Result<std::unique_ptr<IExpression>>;
    auto ParseAnd() -> Result<std::unique_ptr<IExpression>>;
    auto ParseNot() -> Result<std::unique_ptr<IExpression>>;
    auto ParseCompare() -> Result<std::unique_ptr<IExpression>>;
    auto ParseAddSub() -> Result<std::unique_ptr<IExpression>>;
    auto ParseMulDiv() -> Result<std::unique_ptr<IExpression>>;
    auto ParseUnary() -> Result<std::unique_ptr<IExpression>>;
    auto ParseAtom() -> Result<std::unique_ptr<IExpression>>;
    auto ParseFunctionCall(std::string name) -> Result<std::unique_ptr<IExpression>>;
    auto ParseListLiteral() -> Result<std::unique_ptr<IExpression>>;

    // Token stream helpers
    auto Peek() const -> const Token&;
    auto Advance() -> Token;
    auto Expect(TokenType t) -> Result<Token>;
    bool IsAtEnd() const;

    std::vector<Token> tokens_;
    size_t current_;
};

}  // namespace sai::rule
