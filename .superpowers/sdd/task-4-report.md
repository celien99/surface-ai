# Task 4 Report: Lexer

**Status:** Complete

**Commit:** `b3a9f87` (`feat(rule): ✨ 词法分析器（Tokenize 中缀表达式 → Token 流）`)

## Files changed

| File | Action |
|------|--------|
| `src/rule/lexer.h` | **Created** -- Internal header with `TokenType`, `Token`, `Lexer` class |
| `src/rule/lexer.cpp` | **Replaced** -- Full implementation of `Tokenize`, `NextToken`, `SkipWhitespace`, `Match`, `MakeToken`, `LexString`, `LexNumber`, `LexIdentifier` |
| `tests/rule/lexer_test.cpp` | **Replaced** -- 9 GoogleTest test cases |
| `tests/rule/CMakeLists.txt` | **Modified** -- Added `${CMAKE_SOURCE_DIR}` to test include path for `#include "src/rule/lexer.h"` |

## Test summary

```
100% tests passed, 0 tests failed out of 29 (all rule tests)
  - 9 LexerTest cases: all PASS
  - 20 existing rule test cases: all PASS (regression)
```

## Test cases implemented

1. `TokenizeEmpty` -- Empty string returns only End token
2. `TokenizeNumbers` -- Integer `123` and float `3.14` produce Number tokens with correct text
3. `TokenizeStrings` -- Double-quoted `"hello"` and single-quoted `'world'` produce String tokens
4. `TokenizeBools` -- `true`/`false` produce Bool tokens (case-sensitive)
5. `TokenizeIdentifiers` -- `defect.score material.name` produces Identifier + Dot sequences
6. `TokenizeOperators` -- All single/double-char operators and keywords (AND/OR/NOT) produce correct token types
7. `TokenizeArrow` -- `material->supplier->batch.reject_rate` correctly handles Arrow vs Minus
8. `TokenizeParensAndBrackets` -- `(a IN [1, 2])` produces correct parenthesized/braced sequence with IN keyword
9. `UnexpectedCharacterReturnsError` -- `@` produces `ErrorCode::Rule_ParseError`

## Design notes

- **Header location:** Internal header at `src/rule/lexer.h` (not in `include/sai/rule/`) since the lexer is an implementation detail of the rule parser.
- **No escape sequences:** String literals do not support escape sequences (consistent with brief which didn't mention them).
- **Keyword matching:** Case-sensitive per brief (`true`, `false`, `AND`, `OR`, `NOT`, `IN`). `True`/`and` etc. are treated as identifiers.
- **Number leading `"."`:** `.5` is tokenized as Number (checked in `NextToken` by looking ahead).
- **Two-char precedence:** Two-char operators (`>=`, `<=`, `!=`, `==`, `->`) are checked before single-char operators, so `->` consumes both chars rather than `-` (minus) then `>` (gt).
- **String unterminated error:** If end of source is reached before closing quote, returns `Rule_ParseError`.

## Concerns

None. All 9 LexerTest cases and all 20 pre-existing rule tests pass clean.
