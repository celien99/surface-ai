#include "lexer.h"
#include <cctype>
#include <sstream>

namespace sai::rule {

// --- Token ---
std::string Token::ToString() const {
    switch (type) {
    case TokenType::Number: return "Number(" + text + ")";
    case TokenType::String: return "String(" + text + ")";
    case TokenType::Bool: return "Bool(" + text + ")";
    case TokenType::Identifier: return "Identifier(" + text + ")";
    case TokenType::Arrow: return "Arrow";
    case TokenType::Dot: return "Dot";
    case TokenType::Plus: return "Plus";
    case TokenType::Minus: return "Minus";
    case TokenType::Star: return "Star";
    case TokenType::Slash: return "Slash";
    case TokenType::Eq: return "Eq";
    case TokenType::Neq: return "Neq";
    case TokenType::Lt: return "Lt";
    case TokenType::Le: return "Le";
    case TokenType::Gt: return "Gt";
    case TokenType::Ge: return "Ge";
    case TokenType::And: return "And";
    case TokenType::Or: return "Or";
    case TokenType::Not: return "Not";
    case TokenType::In: return "In";
    case TokenType::LParen: return "LParen";
    case TokenType::RParen: return "RParen";
    case TokenType::LBracket: return "LBracket";
    case TokenType::RBracket: return "RBracket";
    case TokenType::Comma: return "Comma";
    case TokenType::End: return "End";
    }
    return "Unknown";
}

// --- Lexer ---
Lexer::Lexer(std::string_view source) : source_(source), pos_(0), start_(0) {}

auto Lexer::Tokenize() -> Result<std::vector<Token>> {
    std::vector<Token> tokens;
    while (true) {
        auto token = NextToken();
        if (!token.has_value()) return tl::unexpected(token.error());
        tokens.push_back(*token);
        if (token->type == TokenType::End) break;
    }
    return tokens;
}

auto Lexer::NextToken() -> Result<Token> {
    SkipWhitespace();
    start_ = pos_;
    if (pos_ >= source_.size()) return Token{TokenType::End, "", pos_};

    // Two-char operators first
    if (Match(">=")) return MakeToken(TokenType::Ge);
    if (Match("<=")) return MakeToken(TokenType::Le);
    if (Match("!=")) return MakeToken(TokenType::Neq);
    if (Match("==")) return MakeToken(TokenType::Eq);
    if (Match("->")) return MakeToken(TokenType::Arrow);

    // Single-char operators
    char c = source_[pos_];
    if (c == '+') { pos_++; return MakeToken(TokenType::Plus); }
    if (c == '-') { pos_++; return MakeToken(TokenType::Minus); }
    if (c == '*') { pos_++; return MakeToken(TokenType::Star); }
    if (c == '/') { pos_++; return MakeToken(TokenType::Slash); }
    if (c == '>') { pos_++; return MakeToken(TokenType::Gt); }
    if (c == '<') { pos_++; return MakeToken(TokenType::Lt); }
    if (c == '(') { pos_++; return MakeToken(TokenType::LParen); }
    if (c == ')') { pos_++; return MakeToken(TokenType::RParen); }
    if (c == '[') { pos_++; return MakeToken(TokenType::LBracket); }
    if (c == ']') { pos_++; return MakeToken(TokenType::RBracket); }
    if (c == ',') { pos_++; return MakeToken(TokenType::Comma); }
    if (c == '.') { pos_++; return MakeToken(TokenType::Dot); }

    // Strings
    if (c == '"' || c == '\'') return LexString(c);

    // Numbers (including leading '.' like ".5")
    if (std::isdigit(static_cast<unsigned char>(c)) ||
        (c == '.' && pos_ + 1 < source_.size() &&
         std::isdigit(static_cast<unsigned char>(source_[pos_ + 1]))))
        return LexNumber();

    // Identifiers and keywords
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') return LexIdentifier();

    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Rule_ParseError,
        std::string("Unexpected character: '") + c + "' at position " + std::to_string(pos_),
        std::source_location::current(),
    });
}

void Lexer::SkipWhitespace() {
    while (pos_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[pos_])))
        pos_++;
}

bool Lexer::Match(std::string_view prefix) {
    if (pos_ + prefix.size() <= source_.size() &&
        source_.substr(pos_, prefix.size()) == prefix) {
        pos_ += prefix.size();
        return true;
    }
    return false;
}

Token Lexer::MakeToken(TokenType t) {
    return Token{t, std::string(source_.substr(start_, pos_ - start_)), start_};
}

auto Lexer::LexString(char quote) -> Result<Token> {
    size_t token_pos = pos_;  // position of opening quote
    pos_++;                   // skip opening quote
    while (pos_ < source_.size() && source_[pos_] != quote)
        pos_++;
    if (pos_ >= source_.size()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            std::string("Unterminated string literal starting at position ")
                + std::to_string(token_pos),
            std::source_location::current(),
        });
    }
    // Extract content between quotes
    std::string content(source_.substr(token_pos + 1, pos_ - token_pos - 1));
    pos_++;  // skip closing quote
    return Token{TokenType::String, content, token_pos};
}

auto Lexer::LexNumber() -> Result<Token> {
    // Consume integer part
    while (pos_ < source_.size() &&
           std::isdigit(static_cast<unsigned char>(source_[pos_])))
        pos_++;
    // Consume fractional part if dot is followed by a digit
    if (pos_ < source_.size() && source_[pos_] == '.' &&
        pos_ + 1 < source_.size() &&
        std::isdigit(static_cast<unsigned char>(source_[pos_ + 1]))) {
        pos_++;  // consume the dot
        while (pos_ < source_.size() &&
               std::isdigit(static_cast<unsigned char>(source_[pos_])))
            pos_++;
    }
    return MakeToken(TokenType::Number);
}

auto Lexer::LexIdentifier() -> Result<Token> {
    while (pos_ < source_.size() &&
           (std::isalnum(static_cast<unsigned char>(source_[pos_])) ||
            source_[pos_] == '_'))
        pos_++;
    Token tok = MakeToken(TokenType::Identifier);
    // Check for case-sensitive keywords
    if (tok.text == "true" || tok.text == "false")
        tok.type = TokenType::Bool;
    else if (tok.text == "AND")
        tok.type = TokenType::And;
    else if (tok.text == "OR")
        tok.type = TokenType::Or;
    else if (tok.text == "NOT")
        tok.type = TokenType::Not;
    else if (tok.text == "IN")
        tok.type = TokenType::In;
    return tok;
}

}  // namespace sai::rule
