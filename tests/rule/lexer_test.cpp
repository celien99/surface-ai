#include <gtest/gtest.h>

#include "src/rule/lexer.h"

namespace sai::rule {
namespace {

TEST(LexerTest, TokenizeEmpty) {
    Lexer lex("");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    ASSERT_EQ(tokens->size(), 1);
    EXPECT_EQ((*tokens)[0].type, TokenType::End);
}

TEST(LexerTest, TokenizeNumbers) {
    Lexer lex("123 3.14");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    ASSERT_EQ(tokens->size(), 3);  // 123, 3.14, End
    EXPECT_EQ((*tokens)[0].type, TokenType::Number);
    EXPECT_EQ((*tokens)[0].text, "123");
    EXPECT_EQ((*tokens)[1].type, TokenType::Number);
    EXPECT_EQ((*tokens)[1].text, "3.14");
}

TEST(LexerTest, TokenizeStrings) {
    Lexer lex(R"("hello" 'world')");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    EXPECT_EQ((*tokens)[0].type, TokenType::String);
    EXPECT_EQ((*tokens)[0].text, "hello");
    EXPECT_EQ((*tokens)[1].type, TokenType::String);
    EXPECT_EQ((*tokens)[1].text, "world");
}

TEST(LexerTest, TokenizeBools) {
    Lexer lex("true false");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    EXPECT_EQ((*tokens)[0].type, TokenType::Bool);
    EXPECT_EQ((*tokens)[0].text, "true");
    EXPECT_EQ((*tokens)[1].type, TokenType::Bool);
    EXPECT_EQ((*tokens)[1].text, "false");
}

TEST(LexerTest, TokenizeIdentifiers) {
    Lexer lex("defect.score material.name");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    // defect . score material . name End
    EXPECT_EQ((*tokens)[0].text, "defect");
    EXPECT_EQ((*tokens)[1].type, TokenType::Dot);
    EXPECT_EQ((*tokens)[2].text, "score");
}

TEST(LexerTest, TokenizeOperators) {
    Lexer lex("+ - * / == != > < >= <= AND OR NOT");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    EXPECT_EQ((*tokens)[0].type, TokenType::Plus);
    EXPECT_EQ((*tokens)[4].type, TokenType::Eq);
    EXPECT_EQ((*tokens)[5].type, TokenType::Neq);
    EXPECT_EQ((*tokens)[8].type, TokenType::Ge);
    EXPECT_EQ((*tokens)[9].type, TokenType::Le);
    EXPECT_EQ((*tokens)[10].type, TokenType::And);
    EXPECT_EQ((*tokens)[11].type, TokenType::Or);
    EXPECT_EQ((*tokens)[12].type, TokenType::Not);
}

TEST(LexerTest, TokenizeArrow) {
    Lexer lex("material->supplier->batch.reject_rate");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    EXPECT_EQ((*tokens)[0].text, "material");
    EXPECT_EQ((*tokens)[1].type, TokenType::Arrow);
    EXPECT_EQ((*tokens)[2].text, "supplier");
    EXPECT_EQ((*tokens)[3].type, TokenType::Arrow);
}

TEST(LexerTest, TokenizeParensAndBrackets) {
    Lexer lex("(a IN [1, 2])");
    auto tokens = lex.Tokenize();
    ASSERT_TRUE(tokens.has_value());
    EXPECT_EQ((*tokens)[0].type, TokenType::LParen);
    EXPECT_EQ((*tokens)[3].type, TokenType::LBracket);
    EXPECT_EQ((*tokens)[5].type, TokenType::Comma);
    EXPECT_EQ((*tokens)[7].type, TokenType::RBracket);
    EXPECT_EQ((*tokens)[8].type, TokenType::RParen);
}

TEST(LexerTest, UnexpectedCharacterReturnsError) {
    Lexer lex("defect @ score");
    auto tokens = lex.Tokenize();
    EXPECT_FALSE(tokens.has_value());
    EXPECT_EQ(tokens.error().code, ErrorCode::Rule_ParseError);
}

}  // namespace
}  // namespace sai::rule
