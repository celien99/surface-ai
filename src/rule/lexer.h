#pragma once

#include <string>
#include <string_view>
#include <vector>
#include "sai/core/error.h"

namespace sai::rule {

enum class TokenType {
    Number, String, Bool,
    Identifier,
    Arrow, Dot,
    Plus, Minus, Star, Slash,
    Eq, Neq, Lt, Le, Gt, Ge,
    And, Or, Not, In,
    LParen, RParen, LBracket, RBracket,
    Comma,
    End
};

struct Token {
    TokenType type;
    std::string text;
    size_t position;
    std::string ToString() const;
};

class Lexer {
public:
    explicit Lexer(std::string_view source);
    auto Tokenize() -> Result<std::vector<Token>>;
private:
    auto NextToken() -> Result<Token>;
    void SkipWhitespace();
    bool Match(std::string_view prefix);
    Token MakeToken(TokenType t);
    auto LexString(char quote) -> Result<Token>;
    auto LexNumber() -> Result<Token>;
    auto LexIdentifier() -> Result<Token>;

    std::string_view source_;
    size_t pos_;
    size_t start_;  // start of current token within source_
};

}  // namespace sai::rule
