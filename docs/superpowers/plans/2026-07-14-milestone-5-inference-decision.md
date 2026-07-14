# Milestone 5 推理决策 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rule Engine (YAML-driven expression AST + FactBase materialization + conflict resolution) and Reasoner (configurable decision tree + weighted sigmoid scoring + operator-level trace + full-chain evidence).

**Architecture:** FactBase materialization layer bridges M3 Detection/M4 Knowledge into a flat key-value table consumed by both the Rule Engine (AST interpretation with memoization + short-circuit evaluation) and the Reasoner (recursive decision tree walk + multi-formula max aggregation). Rule sets evaluate in parallel on WorkerPool; single-rule failures skip without halting the pipeline.

**Tech Stack:** C++20, yaml-cpp, spdlog, GoogleTest, WorkerPool (M1), KnowledgeGraph (M4), VectorPath (M4), DetectionResult (M3)

## Global Constraints

- All public types in `sai::rule` (5.1) or `sai::reasoner` (5.2) namespaces — never `sai::` directly
- Error handling via `Result<T>` (tl::expected), exceptions only for construction failures
- Error codes append-only, never reorder, never touch other batches' members
- Headers use `#pragma once`, include paths mirror module structure
- CUDA-gated and Linux-gated code excluded from local macOS build
- Single-rule evaluation failure → skip rule, do NOT halt the pipeline
- Default branch MUST exist on every BranchNode — incomplete tree rejected at load time
- Spec: `docs/superpowers/specs/2026-07-14-milestone-5-inference-decision-design.md`

---

### Task 1: CMake scaffold + ErrorCode append

**Files:**
- Modify: `CMakeLists.txt:43-48` (add rule and reasoner subdirectories)
- Modify: `include/sai/core/error.h:61-63` (append M5 error codes)
- Create: `src/rule/CMakeLists.txt`
- Create: `src/reasoner/CMakeLists.txt`
- Create: `tests/rule/CMakeLists.txt`
- Create: `tests/reasoner/CMakeLists.txt`

**Interfaces:**
- Produces: `sai::rule` (static lib), `sai::reasoner` (static lib), 8 new ErrorCode members

- [ ] **Step 1: Append M5 error codes to error.h**

Insert after `Retrieval_FusionConfigInvalid,`:

```cpp
    // Rule (M5)
    Rule_ParseError,
    Rule_InvalidPath,
    Rule_TypeMismatch,
    Rule_CyclicOverride,
    Rule_ReloadFailed,
    // Reasoner (M5)
    Reasoner_TreeLoadFailed,
    Reasoner_InvalidTree,
    Reasoner_ScoreComputationFailed,
};
```

- [ ] **Step 2: Create src/rule/CMakeLists.txt**

```cmake
find_package(yaml-cpp CONFIG REQUIRED)

set(SAI_RULE_SOURCES
    value.cpp
    lexer.cpp
    parser.cpp
    expression.cpp
    fact_base.cpp
    fact_builder.cpp
    rule_engine.cpp
    conflict_resolver.cpp
)

add_library(sai_rule STATIC ${SAI_RULE_SOURCES})
add_library(sai::rule ALIAS sai_rule)
target_include_directories(sai_rule PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_rule PUBLIC sai::core sai::knowledge sai::retrieval sai::detection yaml-cpp::yaml-cpp)
target_compile_features(sai_rule PUBLIC cxx_std_20)
```

- [ ] **Step 3: Create src/reasoner/CMakeLists.txt**

```cmake
find_package(yaml-cpp CONFIG REQUIRED)

set(SAI_REASONER_SOURCES
    decision_tree.cpp
    tree_walker.cpp
    score_calculator.cpp
    trace_recorder.cpp
    evidence_collector.cpp
    reasoner.cpp
)

add_library(sai_reasoner STATIC ${SAI_REASONER_SOURCES})
add_library(sai::reasoner ALIAS sai_reasoner)
target_include_directories(sai_reasoner PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_reasoner PUBLIC sai::core sai::rule yaml-cpp::yaml-cpp)
target_compile_features(sai_reasoner PUBLIC cxx_std_20)
```

- [ ] **Step 4: Create tests/rule/CMakeLists.txt**

```cmake
find_package(GTest CONFIG REQUIRED)

set(SAI_RULE_TEST_SOURCES
    value_test.cpp
    lexer_test.cpp
    parser_test.cpp
    expression_test.cpp
    fact_base_test.cpp
    fact_builder_test.cpp
    rule_engine_test.cpp
    conflict_resolver_test.cpp
    integration_test.cpp
)

add_executable(sai_rule_tests ${SAI_RULE_TEST_SOURCES})
target_include_directories(sai_rule_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_rule_tests PRIVATE sai::rule GTest::gtest GTest::gtest_main)
target_compile_features(sai_rule_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_rule_tests)
```

- [ ] **Step 5: Create tests/reasoner/CMakeLists.txt**

```cmake
find_package(GTest CONFIG REQUIRED)

set(SAI_REASONER_TEST_SOURCES
    decision_tree_test.cpp
    score_calculator_test.cpp
    reasoner_test.cpp
    integration_test.cpp
)

add_executable(sai_reasoner_tests ${SAI_REASONER_TEST_SOURCES})
target_include_directories(sai_reasoner_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_reasoner_tests PRIVATE sai::reasoner GTest::gtest GTest::gtest_main)
target_compile_features(sai_reasoner_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_reasoner_tests)
```

- [ ] **Step 6: Register subdirectories in root CMakeLists.txt**

Add after `add_subdirectory(tests/retrieval)`:

```cmake
add_subdirectory(src/rule)
add_subdirectory(tests/rule)
add_subdirectory(src/reasoner)
add_subdirectory(tests/reasoner)
```

- [ ] **Step 7: Create empty placeholder .cpp files so CMake configures**

Create each .cpp file from the SAI_RULE_SOURCES and SAI_REASONER_SOURCES lists as an empty file with just `#include "sai/rule/..."` or `#include "sai/reasoner/..."` matching its header.

- [ ] **Step 8: Create minimal public headers with namespace guards**

For each header in `include/sai/rule/` and `include/sai/reasoner/`, create a minimal file:

```cpp
// include/sai/rule/value.h (example)
#pragma once

namespace sai::rule {
// Task 2 will fill this in
}  // namespace sai::rule
```

- [ ] **Step 9: Build to verify CMake configuration**

```bash
cmake --preset default && cmake --build --preset default
```

Expected: Configures and builds without errors (libraries have empty or minimal content).

- [ ] **Step 10: Commit**

```bash
git add CMakeLists.txt include/sai/core/error.h src/rule/ src/reasoner/ tests/rule/ tests/reasoner/ include/sai/rule/ include/sai/reasoner/
git commit -m "chore(rule,reasoner): 🔧 M5 CMake 骨架 + ErrorCode 追加（Rule_*/Reasoner_*）"
```

---

### Task 2: Value type system

**Files:**
- Create: `include/sai/rule/value.h`
- Create: `src/rule/value.cpp`
- Create: `tests/rule/value_test.cpp`

**Interfaces:**
- Produces: `sai::rule::Value` (Kind: Null/Scalar/List), `sai::rule::Scalar` (variant<double, string, bool>)

- [ ] **Step 1: Write the header**

```cpp
// include/sai/rule/value.h
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sai::rule {

using Scalar = std::variant<double, std::string, bool>;

class Value {
public:
    enum class Kind { Null, Scalar, List };

    static auto Null() -> Value;
    static auto Of(double v) -> Value;
    static auto Of(std::string v) -> Value;
    static auto Of(bool v) -> Value;
    static auto OfList(std::vector<Value> v) -> Value;

    auto GetKind() const noexcept -> Kind;
    auto IsNull() const -> bool;

    auto AsDouble() const -> std::optional<double>;
    auto AsString() const -> std::optional<std::string_view>;
    auto AsBool() const -> std::optional<bool>;
    auto AsList() const -> std::optional<const std::vector<Value>&>;

    auto operator+(const Value& rhs) const -> Value;
    auto operator-(const Value& rhs) const -> Value;
    auto operator*(const Value& rhs) const -> Value;
    auto operator/(const Value& rhs) const -> Value;

    auto operator==(const Value& rhs) const -> bool;
    auto operator<(const Value& rhs) const -> bool;
    auto operator>(const Value& rhs) const -> bool;

private:
    Kind kind_{Kind::Null};
    std::optional<Scalar> scalar_;
    std::optional<std::vector<Value>> list_;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write failing tests**

```cpp
// tests/rule/value_test.cpp
#include <sai/rule/value.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(ValueTest, NullByDefault) {
    Value v;
    EXPECT_EQ(v.GetKind(), Value::Kind::Null);
    EXPECT_TRUE(v.IsNull());
}

TEST(ValueTest, OfDoubleRoundTrips) {
    Value v = Value::Of(3.14);
    EXPECT_EQ(v.GetKind(), Value::Kind::Scalar);
    EXPECT_FALSE(v.IsNull());
    auto d = v.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 3.14);
}

TEST(ValueTest, OfStringRoundTrips) {
    Value v = Value::Of(std::string("hello"));
    EXPECT_EQ(v.GetKind(), Value::Kind::Scalar);
    auto s = v.AsString();
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(*s, "hello");
}

TEST(ValueTest, OfBoolRoundTrips) {
    Value v = Value::Of(true);
    auto b = v.AsBool();
    ASSERT_TRUE(b.has_value());
    EXPECT_TRUE(*b);
}

TEST(ValueTest, OfListRoundTrips) {
    auto v = Value::OfList({Value::Of(1.0), Value::Of(2.0), Value::Of(3.0)});
    EXPECT_EQ(v.GetKind(), Value::Kind::List);
    auto lst = v.AsList();
    ASSERT_TRUE(lst.has_value());
    EXPECT_EQ(lst->size(), 3);
}

TEST(ValueTest, TypeMismatchReturnsNullopt) {
    Value v = Value::Of(42.0);
    EXPECT_FALSE(v.AsString().has_value());
    EXPECT_FALSE(v.AsBool().has_value());
    EXPECT_FALSE(v.AsList().has_value());
}

TEST(ValueTest, AddDoubles) {
    Value a = Value::Of(3.0);
    Value b = Value::Of(2.0);
    Value c = a + b;
    auto d = c.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 5.0);
}

TEST(ValueTest, AddWithNullReturnsNull) {
    Value a = Value::Null();
    Value b = Value::Of(2.0);
    Value c = a + b;
    EXPECT_TRUE(c.IsNull());
}

TEST(ValueTest, MulDoubles) {
    Value a = Value::Of(3.0);
    Value b = Value::Of(2.0);
    Value c = a * b;
    auto d = c.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 6.0);
}

TEST(ValueTest, DivByZeroReturnsNull) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(0.0);
    Value c = a / b;
    EXPECT_TRUE(c.IsNull());
}

TEST(ValueTest, StringEquality) {
    Value a = Value::Of(std::string("abc"));
    Value b = Value::Of(std::string("abc"));
    Value c = Value::Of(std::string("xyz"));
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(ValueTest, NumericComparison) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(3.0);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(a > b);
}

TEST(ValueTest, CrossTypeComparisonReturnsFalse) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(std::string("hello"));
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a > b);
}

}  // namespace
}  // namespace sai::rule
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "ValueTest" --output-on-failure
```

Expected: Build succeeds (if CMake configured), tests FAIL because Value methods are not implemented.

- [ ] **Step 4: Implement value.cpp**

```cpp
// src/rule/value.cpp
#include "sai/rule/value.h"
#include <cmath>
#include <stdexcept>

namespace sai::rule {

auto Value::Null() -> Value { return Value{}; }

auto Value::Of(double v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{v};
    return val;
}

auto Value::Of(std::string v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{std::move(v)};
    return val;
}

auto Value::Of(bool v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{v};
    return val;
}

auto Value::OfList(std::vector<Value> v) -> Value {
    Value val;
    val.kind_ = Kind::List;
    val.list_ = std::move(v);
    return val;
}

auto Value::GetKind() const noexcept -> Kind { return kind_; }
auto Value::IsNull() const -> bool { return kind_ == Kind::Null; }

auto Value::AsDouble() const -> std::optional<double> {
    if (!scalar_.has_value()) return std::nullopt;
    return std::get_if<double>(&*scalar_) ? std::optional{std::get<double>(*scalar_)} : std::nullopt;
}

auto Value::AsString() const -> std::optional<std::string_view> {
    if (!scalar_.has_value()) return std::nullopt;
    auto* s = std::get_if<std::string>(&*scalar_);
    return s ? std::optional{std::string_view{*s}} : std::nullopt;
}

auto Value::AsBool() const -> std::optional<bool> {
    if (!scalar_.has_value()) return std::nullopt;
    auto* b = std::get_if<bool>(&*scalar_);
    return b ? std::optional{*b} : std::nullopt;
}

auto Value::AsList() const -> std::optional<const std::vector<Value>&> {
    if (!list_.has_value()) return std::nullopt;
    return *list_;
}

namespace {
auto AsNumeric(const Value& v) -> std::optional<double> {
    if (auto d = v.AsDouble()) return d;
    if (auto b = v.AsBool()) return *b ? 1.0 : 0.0;
    return std::nullopt;
}
}  // namespace

auto Value::operator+(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a + *b);
}

auto Value::operator-(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a - *b);
}

auto Value::operator*(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a * *b);
}

auto Value::operator/(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b || *b == 0.0) return Null();
    return Of(*a / *b);
}

auto Value::operator==(const Value& rhs) const -> bool {
    if (kind_ != rhs.kind_) return false;
    if (kind_ == Kind::Null) return true;
    if (kind_ == Kind::Scalar) return scalar_ == rhs.scalar_;
    // List comparison: size + element-wise
    if (!list_ || !rhs.list_) return false;
    if (list_->size() != rhs.list_->size()) return false;
    for (size_t i = 0; i < list_->size(); ++i) {
        if (!((*list_)[i] == (*rhs.list_)[i])) return false;
    }
    return true;
}

auto Value::operator<(const Value& rhs) const -> bool {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return false;
    return *a < *b;
}

auto Value::operator>(const Value& rhs) const -> bool {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return false;
    return *a > *b;
}

}  // namespace sai::rule
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "ValueTest" --output-on-failure
```

Expected: All 13 ValueTest cases PASS.

- [ ] **Step 6: Commit**

```bash
git add include/sai/rule/value.h src/rule/value.cpp tests/rule/value_test.cpp
git commit -m "feat(rule): ✨ Value 类型系统（Null/Scalar/List + 算术 + 比较）"
```

---

### Task 3: Expression AST interfaces + concrete node types

**Files:**
- Create: `include/sai/rule/expression.h`
- Create: `include/sai/rule/ast_nodes.h`

**Interfaces:**
- Produces: `IExpression`, `ExprKind`, `BinaryOp`, `UnaryOp`, `BuiltinFn`, `LiteralExpr`, `FieldRefExpr`, `BinaryExpr`, `UnaryExpr`, `FunctionExpr`, `PathExpr`

- [ ] **Step 1: Write expression.h (interfaces + enums)**

```cpp
// include/sai/rule/expression.h
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "sai/core/object.h"
#include "sai/core/error.h"
#include "sai/rule/value.h"

namespace sai::rule {

enum class ExprKind { Literal, FieldRef, Binary, Unary, Function, Path };
enum class BinaryOp { And, Or, Add, Sub, Mul, Div, Eq, Neq, Lt, Le, Gt, Ge, In };
enum class UnaryOp { Not, Neg };
enum class BuiltinFn { Avg, Count, Max, Min, Sum, Len };

class FactBase;  // forward decl (Task 7)

class IExpression : public core::Object {
public:
    virtual auto GetKind() const noexcept -> ExprKind = 0;
    virtual auto Evaluate(FactBase& ctx) const -> Result<Value> = 0;
    virtual auto CollectFieldRefs() const -> std::vector<std::string> = 0;
    virtual auto SourceText() const -> std::string_view = 0;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write ast_nodes.h (concrete node types)**

```cpp
// include/sai/rule/ast_nodes.h
#pragma once

#include "sai/rule/expression.h"

namespace sai::rule {

class LiteralExpr final : public IExpression {
public:
    explicit LiteralExpr(Value val);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::Literal; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override { return {}; }
    auto SourceText() const -> std::string_view override;
    auto GetValue() const -> const Value&;
private:
    Value value_;
    std::string source_text_;
};

class FieldRefExpr final : public IExpression {
public:
    explicit FieldRefExpr(std::string key);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::FieldRef; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override;
    auto SourceText() const -> std::string_view override;
private:
    std::string key_;
};

class BinaryExpr final : public IExpression {
public:
    BinaryExpr(BinaryOp op, std::unique_ptr<IExpression> lhs, std::unique_ptr<IExpression> rhs, std::string source);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::Binary; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override;
    auto SourceText() const -> std::string_view override;
    auto GetOp() const -> BinaryOp;
    auto GetLhs() const -> const IExpression&;
    auto GetRhs() const -> const IExpression&;
private:
    BinaryOp op_;
    std::unique_ptr<IExpression> lhs_, rhs_;
    std::string source_text_;
};

class UnaryExpr final : public IExpression {
public:
    UnaryExpr(UnaryOp op, std::unique_ptr<IExpression> operand, std::string source);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::Unary; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override;
    auto SourceText() const -> std::string_view override;
private:
    UnaryOp op_;
    std::unique_ptr<IExpression> operand_;
    std::string source_text_;
};

class FunctionExpr final : public IExpression {
public:
    FunctionExpr(BuiltinFn fn, std::vector<std::unique_ptr<IExpression>> args, std::string source);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::Function; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override;
    auto SourceText() const -> std::string_view override;
private:
    BuiltinFn fn_;
    std::vector<std::unique_ptr<IExpression>> args_;
    std::string source_text_;
};

class PathExpr final : public IExpression {
public:
    explicit PathExpr(std::string path);
    auto GetKind() const noexcept -> ExprKind override { return ExprKind::Path; }
    auto Evaluate(FactBase& ctx) const -> Result<Value> override;
    auto CollectFieldRefs() const -> std::vector<std::string> override;
    auto SourceText() const -> std::string_view override;
    auto GetPath() const -> std::string_view;
    auto GetFlattenedKey() const -> std::string_view;
private:
    std::string path_;           // "material->supplier->batch.reject_rate"
    std::string flattened_key_;  // "material.supplier.batch.reject_rate"
};

}  // namespace sai::rule
```

- [ ] **Step 3: Write minimal expression_test.cpp (compile check only for now)**

```cpp
// tests/rule/expression_test.cpp
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
```

- [ ] **Step 4: Implement ast_nodes.cpp constructors + trivial methods**

Create `src/rule/expression.cpp` with constructor implementations for all 6 node types (Evaluate will be fleshed out in Task 6).

- [ ] **Step 5: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "AstNodeTest" --output-on-failure
```

Expected: Builds and all 5 AstNodeTest cases PASS.

- [ ] **Step 6: Commit**

```bash
git add include/sai/rule/expression.h include/sai/rule/ast_nodes.h src/rule/expression.cpp tests/rule/expression_test.cpp
git commit -m "feat(rule): ✨ Expression AST 接口 + 6 种具体节点类型"
```

---

### Task 4: Lexer

**Files:**
- Create: `src/rule/lexer.cpp` (internal, no public header — lexer is an implementation detail of RuleParser)
- Create: `tests/rule/lexer_test.cpp`

**Interfaces:**
- Produces: `sai::rule::Token` (internal), `sai::rule::Lexer` (internal)
- Consumes: Nothing beyond Task 2 (Value)/Task 3 (operators)

- [ ] **Step 1: Write lexer.cpp**

```cpp
// src/rule/lexer.cpp
#include "lexer.h"  // internal header
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
Lexer::Lexer(std::string_view source) : source_(source), pos_(0) {}

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
    
    // Numbers
    if (std::isdigit(c) || (c == '.' && pos_ + 1 < source_.size() && std::isdigit(source_[pos_ + 1])))
        return LexNumber();
    
    // Identifiers and keywords
    if (std::isalpha(c) || c == '_') return LexIdentifier();
    
    return tl::unexpected(ErrorInfo{
        ErrorCode::Rule_ParseError,
        std::string("Unexpected character: '") + c + "' at position " + std::to_string(pos_)
    });
}

// ... (helper methods: SkipWhitespace, Match, LexString, LexNumber, LexIdentifier)
```

- [ ] **Step 2: Write the internal lexer.h header**

```cpp
// src/rule/lexer.h
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
};

}  // namespace sai::rule
```

- [ ] **Step 3: Write lexer_test.cpp**

```cpp
// tests/rule/lexer_test.cpp
#include "../src/rule/lexer.h"  // internal header
#include <gtest/gtest.h>

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
```

- [ ] **Step 4: Build and run tests to verify they pass**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "LexerTest" --output-on-failure
```

Expected: All 9 LexerTest cases PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rule/lexer.h src/rule/lexer.cpp tests/rule/lexer_test.cpp
git commit -m "feat(rule): ✨ 词法分析器（Tokenize 中缀表达式 → Token 流）"
```

---

### Task 5: Parser (recursive descent)

**Files:**
- Create: `src/rule/parser.h` (internal)
- Create: `src/rule/parser.cpp`
- Create: `tests/rule/parser_test.cpp`

**Interfaces:**
- Produces: `sai::rule::Parser` (internal, `Parse() -> Result<unique_ptr<IExpression>>`)
- Consumes: Lexer (Task 4), AST nodes (Task 3)

- [ ] **Step 1: Write parser.h**

```cpp
// src/rule/parser.h
#pragma once

#include <memory>
#include <vector>
#include "sai/core/error.h"
#include "sai/rule/expression.h"
#include "lexer.h"

namespace sai::rule {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);
    auto Parse() -> Result<std::unique_ptr<IExpression>>;
    
private:
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
    
    auto Peek() const -> const Token&;
    auto Advance() -> Token;
    auto Expect(TokenType t) -> Result<Token>;
    bool IsAtEnd() const;
    
    std::vector<Token> tokens_;
    size_t current_;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write parser.cpp**

Implement recursive descent: `ParseOr → ParseAnd → ParseNot → ParseCompare → ParseAddSub → ParseMulDiv → ParseUnary → ParseAtom`.

Key rules:
- `ParseOr`: `ParseAnd (OR ParseAnd)*`
- `ParseAnd`: `ParseNot (AND ParseNot)*`
- `ParseNot`: `NOT? ParseCompare`
- `ParseCompare`: `ParseAddSub ((==|!=|>|<|>=|<=|IN) ParseAddSub)?`
- `ParseAddSub`: `ParseMulDiv ((+|-) ParseMulDiv)*`
- `ParseMulDiv`: `ParseUnary ((*|/) ParseUnary)*`
- `ParseUnary`: `(+|-) ParseAtom`
- `ParseAtom`: Number | String | Bool | Identifier (check for Arrow → PathExpr, Dot → FieldRef chain, LParen → function call) | LParen Expr RParen | LBracket ListLiteral

For `Identifier` with `->`:
```
identifier (-> identifier)* (. identifier)*  → PathExpr or FieldRefExpr
```

- [ ] **Step 3: Write parser_test.cpp**

```cpp
// tests/rule/parser_test.cpp
#include "../src/rule/parser.h"
#include "../src/rule/lexer.h"
#include <sai/rule/ast_nodes.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

auto ParseExpr(std::string_view source) -> std::unique_ptr<IExpression> {
    Lexer lex(source);
    auto tokens = lex.Tokenize();
    EXPECT_TRUE(tokens.has_value());
    Parser parser(std::move(*tokens));
    auto expr = parser.Parse();
    EXPECT_TRUE(expr.has_value());
    return std::move(*expr);
}

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

TEST(ParserTest, ParseArithmetic) {
    auto expr = ParseExpr("defect.score * 2.0 + 1.0");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Add);
}

TEST(ParserTest, ParseComparison) {
    auto expr = ParseExpr("defect.score > 0.8");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Gt);
}

TEST(ParserTest, ParseAndOr) {
    auto expr = ParseExpr("defect.score > 0.8 AND defect.area_mm2 > 10");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::And);
}

TEST(ParserTest, ParseNot) {
    auto expr = ParseExpr("NOT defect.passed");
    EXPECT_EQ(expr->GetKind(), ExprKind::Unary);
}

TEST(ParserTest, ParseInList) {
    auto expr = ParseExpr(R"(defect.position IN ["center", "left"])");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::In);
}

TEST(ParserTest, ParseFunctionCall) {
    auto expr = ParseExpr("AVG(1.0, 2.0, 3.0)");
    EXPECT_EQ(expr->GetKind(), ExprKind::Function);
}

TEST(ParserTest, ParseParenthesized) {
    auto expr = ParseExpr("(defect.score + 1.0) * 2.0");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Mul);
}

TEST(ParserTest, ParsePrecedence) {
    // AND binds tighter than OR: a OR b AND c → a OR (b AND c)
    auto expr = ParseExpr("a OR b AND c");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    EXPECT_EQ(static_cast<BinaryExpr*>(expr.get())->GetOp(), BinaryOp::Or);
    auto rhs = static_cast<BinaryExpr*>(expr.get())->GetRhs();
    EXPECT_EQ(static_cast<const BinaryExpr&>(rhs).GetOp(), BinaryOp::And);
}

TEST(ParserTest, ComplexRuleCondition) {
    auto expr = ParseExpr(R"(defect.type == "scratch" AND defect.area_mm2 > 10 AND defect.position IN ["center", "left"])");
    EXPECT_EQ(expr->GetKind(), ExprKind::Binary);
    // Should be ((type=="scratch" AND area>10) AND position IN [...])
}

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
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "ParserTest" --output-on-failure
```

Expected: All ParserTest tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rule/parser.h src/rule/parser.cpp tests/rule/parser_test.cpp
git commit -m "feat(rule): ✨ 递归下降解析器（Token 流 → AST）"
```

---

### Task 6: AST node evaluation + fact_base integration

**Files:**
- Modify: `src/rule/expression.cpp` (add Evaluate implementations)
- Modify: `tests/rule/expression_test.cpp` (add evaluation tests)

**Interfaces:**
- Consumes: FactBase interface (Task 7 will provide the full impl; here we use a minimal stub)
- Produces: Working `Evaluate()` for all 6 AST node types

- [ ] **Step 1: Extend expression_test.cpp with evaluation tests**

Add tests using a minimal FactBase stub:

```cpp
// Add to tests/rule/expression_test.cpp

// Minimal FactBase stub for AST evaluation tests
class StubFactBase {
public:
    void Set(std::string_view key, Value v) { data_[std::string(key)] = std::move(v); }
    std::optional<Value> Get(std::string_view key) const {
        auto it = data_.find(std::string(key));
        return it != data_.end() ? std::optional{it->second} : std::nullopt;
    }
    bool Has(std::string_view key) const { return data_.count(std::string(key)) > 0; }
private:
    std::unordered_map<std::string, Value> data_;
};

// Note: until Task 7, we use a forward-compatible stub.
// The real FactBase::Get signature is `std::optional<Value> Get(string_view) const`.
```

Wait - this is problematic because `Evaluate(FactBase&)` takes the real FactBase type. Let me handle this differently.

The expression.cpp `Evaluate` implementations need `FactBase&` which doesn't exist yet. Two approaches:
1. Implement expression.cpp Evaluate methods in Task 7 (after FactBase exists)
2. Use a thin wrapper pattern

Given the dependency order, the cleanest approach: implement the Evaluate methods here using only what we know about FactBase from the spec (it has `Get`, `Has`, `Set`). We include `fact_base.h` which at this point will be a forward declaration stub. Then when Task 7 fleshes out FactBase, these Evaluate implementations will compile against the real interface.

Actually, let me restructure: put the Evaluate implementation AFTER FactBase in task order. Task 6 can cover only constructors and trivial methods, and Task 7's FactBase + Task 6's Evaluate can be merged.

Better approach: Task 6 implements `Evaluate` against the real `FactBase` from Task 7, but since they're separate tasks we need a buildable intermediate state. Let me:

1. Task 7 (FactBase) comes BEFORE Task 6 (Evaluate implementations)
2. OR: In Task 6, implement Evaluate against what we know FactBase will look like, using `fact_base.h` header directly

Let me reorder: actually looking at the spec, FactBase is needed by expression evaluation. So let me put FactBase (Task 7) before AST evaluation (Task 6 evaluator part). 

Actually wait - let me think about this more carefully. The cleanest approach is:

Task 6: Implement FactBase + FactSource (simpler, no dependency on AST)
Task 7: Implement Evaluate methods for AST nodes (depends on FactBase)

Let me reorder accordingly.

- [ ] **Step 1: Write the modified task ordering note**

Actually, let me just change my plan. Task 6 becomes FactBase, Task 7 becomes AST evaluation. Let me just write the plan in the correct order rather than fixing it retroactively. I'll write Task 6 as FactBase.

---

### Task 6: FactBase + FactSource

**Files:**
- Create: `include/sai/rule/fact_base.h`
- Create: `src/rule/fact_base.cpp`
- Create: `tests/rule/fact_base_test.cpp`

**Interfaces:**
- Produces: `FactBase`, `FactSource`, `FactSourceKind`
- Consumes: Value (Task 2)

- [ ] **Step 1: Write fact_base.h**

```cpp
// include/sai/rule/fact_base.h
#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "sai/rule/value.h"

namespace sai::rule {

enum class FactSourceKind {
    Direct,
    GraphPath,
    VectorSearch,
    Computed,
    Default
};

struct FactSource {
    FactSourceKind kind;
    std::string description;
    std::chrono::microseconds elapsed{0};
    std::optional<std::string> sql;
    std::optional<int> top_k;
};

class FactBase {
public:
    auto Set(std::string_view key, Value value, FactSource source) -> void;
    auto SetDefault(std::string_view key, Value value) -> void;

    auto Get(std::string_view key) const -> std::optional<Value>;
    auto GetOr(std::string_view key, Value default_val) const -> Value;
    auto Has(std::string_view key) const -> bool;

    auto AddPathMapping(std::string_view graph_path, std::string_view flat_key) -> void;
    auto ResolvePath(std::string_view graph_path) const -> std::optional<std::string>;

    auto SourceOf(std::string_view key) const -> const FactSource&;
    auto AllEntries() const -> std::vector<std::pair<std::string, Value>>;
    auto AllSources() const -> std::vector<std::pair<std::string, FactSource>>;

    auto Snapshot() const -> FactBase;

private:
    struct Entry {
        Value value;
        FactSource source;
    };
    std::map<std::string, Entry, std::less<>> entries_;
    std::map<std::string, std::string, std::less<>> path_mappings_;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write failing tests**

```cpp
// tests/rule/fact_base_test.cpp
#include <sai/rule/fact_base.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(FactBaseTest, SetAndGet) {
    FactBase fb;
    fb.Set("defect.score", Value::Of(0.92), FactSource{FactSourceKind::Direct, "from detection"});
    auto v = fb.Get("defect.score");
    ASSERT_TRUE(v.has_value());
    EXPECT_DOUBLE_EQ(v->AsDouble().value(), 0.92);
}

TEST(FactBaseTest, GetMissingKeyReturnsNullopt) {
    FactBase fb;
    EXPECT_FALSE(fb.Get("nonexistent").has_value());
}

TEST(FactBaseTest, GetOrReturnsDefault) {
    FactBase fb;
    auto v = fb.GetOr("missing", Value::Of(0.5));
    EXPECT_DOUBLE_EQ(v.AsDouble().value(), 0.5);
}

TEST(FactBaseTest, HasKey) {
    FactBase fb;
    fb.Set("key", Value::Of(true), FactSource{});
    EXPECT_TRUE(fb.Has("key"));
    EXPECT_FALSE(fb.Has("other"));
}

TEST(FactBaseTest, SourceTracking) {
    FactBase fb;
    FactSource src{FactSourceKind::GraphPath, "Material→Supplier→Batch", std::chrono::microseconds{1200}, "SELECT ..."};
    fb.Set("material.supplier.name", Value::Of(std::string("SupplierA")), src);
    auto& tracked = fb.SourceOf("material.supplier.name");
    EXPECT_EQ(tracked.kind, FactSourceKind::GraphPath);
    EXPECT_EQ(tracked.sql.value(), "SELECT ...");
}

TEST(FactBaseTest, PathMapping) {
    FactBase fb;
    fb.AddPathMapping("material->supplier->batch.reject_rate", "material.supplier.batch.reject_rate");
    auto flat = fb.ResolvePath("material->supplier->batch.reject_rate");
    ASSERT_TRUE(flat.has_value());
    EXPECT_EQ(*flat, "material.supplier.batch.reject_rate");
}

TEST(FactBaseTest, SnapshotIsIndependent) {
    FactBase fb;
    fb.Set("a", Value::Of(1.0), FactSource{});
    auto snap = fb.Snapshot();
    fb.Set("b", Value::Of(2.0), FactSource{});
    EXPECT_TRUE(fb.Has("b"));
    EXPECT_FALSE(snap.Has("b"));  // snapshot doesn't see later writes
}

TEST(FactBaseTest, AllEntries) {
    FactBase fb;
    fb.Set("x", Value::Of(1.0), FactSource{});
    fb.Set("y", Value::Of(2.0), FactSource{});
    auto entries = fb.AllEntries();
    EXPECT_EQ(entries.size(), 2);
}

TEST(FactBaseTest, SetDefaultOnlyIfMissing) {
    FactBase fb;
    fb.SetDefault("a", Value::Of(10.0));
    fb.Set("a", Value::Of(20.0), FactSource{});
    fb.SetDefault("a", Value::Of(30.0));  // should NOT overwrite
    EXPECT_DOUBLE_EQ(fb.Get("a")->AsDouble().value(), 20.0);
}

}  // namespace
}  // namespace sai::rule
```

- [ ] **Step 3: Implement fact_base.cpp**

Implement all methods. Key: `std::map<std::string, Entry, std::less<>>` with heterogeneous lookup for `string_view` keys.

- [ ] **Step 4: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "FactBaseTest" --output-on-failure
```

Expected: All 9 FactBaseTest cases PASS.

- [ ] **Step 5: Commit**

```bash
git add include/sai/rule/fact_base.h src/rule/fact_base.cpp tests/rule/fact_base_test.cpp
git commit -m "feat(rule): ✨ FactBase 扁平键值表 + FactSource 溯源元数据"
```

---

### Task 7: AST node Evaluate implementations

**Files:**
- Modify: `src/rule/expression.cpp` (add full Evaluate implementations)
- Modify: `tests/rule/expression_test.cpp` (add evaluation tests using real FactBase)

**Interfaces:**
- Consumes: FactBase (Task 6)
- Produces: Working `Evaluate(FactBase&)` for all 6 AST node types + Memoization

- [ ] **Step 1: Extend expression_test.cpp with evaluation tests**

```cpp
// Add to tests/rule/expression_test.cpp

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

TEST(ExpressionEvalTest, ComparisonGt) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(5.0));
    auto rhs = std::make_unique<LiteralExpr>(Value::Of(3.0));
    BinaryExpr expr(BinaryOp::Gt, std::move(lhs), std::move(rhs), "5 > 3");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
}

TEST(ExpressionEvalTest, AndShortCircuits) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(false));
    // rhs references missing key — should not be evaluated
    auto rhs = std::make_unique<FieldRefExpr>("nonexistent");
    BinaryExpr expr(BinaryOp::And, std::move(lhs), std::move(rhs), "false AND nonexistent");
    auto r = expr.Evaluate(fb);
    // Short-circuit means rhs is never evaluated, so missing key doesn't error
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->AsBool().value());
}

TEST(ExpressionEvalTest, OrShortCircuits) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(true));
    auto rhs = std::make_unique<FieldRefExpr>("nonexistent");
    BinaryExpr expr(BinaryOp::Or, std::move(lhs), std::move(rhs), "true OR nonexistent");
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

TEST(ExpressionEvalTest, InOperator) {
    FactBase fb;
    auto lhs = std::make_unique<LiteralExpr>(Value::Of(std::string("center")));
    auto rhs = std::make_unique<LiteralExpr>(Value::OfList({
        Value::Of(std::string("center")),
        Value::Of(std::string("left"))
    }));
    BinaryExpr expr(BinaryOp::In, std::move(lhs), std::move(rhs), R"(center IN [center, left])");
    auto r = expr.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->AsBool().value());
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

TEST(ExpressionEvalTest, PathExprReadsFlattenedKey) {
    FactBase fb;
    fb.AddPathMapping("material->supplier->batch.reject_rate", "material.supplier.batch.reject_rate");
    fb.Set("material.supplier.batch.reject_rate", Value::Of(0.032), FactSource{});
    PathExpr path("material->supplier->batch.reject_rate");
    auto r = path.Evaluate(fb);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->AsDouble().value(), 0.032);
}

TEST(ExpressionEvalTest, MemoizationCachesResult) {
    FactBase fb;
    // First evaluation: field ref reads from FactBase
    fb.Set("defect.score", Value::Of(0.5), FactSource{});
    FieldRefExpr ref("defect.score");
    auto r1 = ref.Evaluate(fb);
    EXPECT_DOUBLE_EQ(r1->AsDouble().value(), 0.5);
    // Memoization: cached result should exist now
    EXPECT_TRUE(fb.Has("__memo__"));  // implementation-defined key prefix
}
```

- [ ] **Step 2: Implement Evaluate for all AST node types in expression.cpp**

Key implementation details:
- `LiteralExpr::Evaluate`: return value_ directly
- `FieldRefExpr::Evaluate`: `fb.Get(key_)` → return value if found, else `Rule_TypeMismatch`
- `BinaryExpr::Evaluate`: dispatch by op, with short-circuit for And/Or
- `UnaryExpr::Evaluate`: Not → `!operand`, Neg → `-operand`
- `FunctionExpr::Evaluate`: dispatch by BuiltinFn
- `PathExpr::Evaluate`: resolve path mapping, then `fb.Get(flattened_key_)`
- Memoization: `Evaluate` checks `fb.Has(hash_of_source)`, returns cached if present; otherwise computes + `fb.Set(hash, result, FactSource{Computed,...})`

- [ ] **Step 3: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "ExpressionEvalTest" --output-on-failure
```

Expected: All ExpressionEvalTest tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rule/expression.cpp tests/rule/expression_test.cpp
git commit -m "feat(rule): ✨ AST 节点 Evaluate 实现（Memoization + 短路求值 + 6 种运算符）"
```

---

### Task 8: FactBuilder

**Files:**
- Create: `include/sai/rule/fact_builder.h`
- Create: `src/rule/fact_builder.cpp`
- Create: `tests/rule/fact_builder_test.cpp`

**Interfaces:**
- Produces: `FactBuilder` (Build method)
- Consumes: FactBase (Task 6), KnowledgeGraph (M4), VectorPath (M4), DetectionResult (M3)

- [ ] **Step 1: Write fact_builder.h**

```cpp
// include/sai/rule/fact_builder.h
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "sai/core/error.h"
#include "sai/rule/fact_base.h"

namespace sai::detection { struct DetectionResult; }
namespace sai::knowledge { class KnowledgeGraph; }
namespace sai::retrieval { class VectorPath; }

namespace sai::rule {

class FactBuilder {
public:
    explicit FactBuilder(
        std::shared_ptr<knowledge::KnowledgeGraph> kg,
        std::shared_ptr<retrieval::VectorPath> vp
    );

    auto Build(
        std::string_view surface_id,
        const detection::DetectionResult& detection,
        const std::vector<std::string>& graph_paths_to_resolve
    ) -> Result<FactBase>;

private:
    auto MapDetection(FactBase&, const detection::DetectionResult&) -> void;
    auto ResolveGraphPaths(FactBase&, const std::vector<std::string>&) -> Result<void>;
    auto RunVectorRetrieval(FactBase&, std::string_view embedding_key) -> Result<void>;

    std::shared_ptr<knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<retrieval::VectorPath> vp_;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write fact_builder_test.cpp**

Since FactBuilder depends on real M3/M4 interfaces, use mocked DetectionResult:
- Test `MapDetection` extracts `defect.type`, `defect.score`, `defect.confidence`, `defect.area_mm2`, `defect.position`, `defect.passed`
- Test `ResolveGraphPaths` with a mock KnowledgeGraph that returns predefined paths
- Test full `Build` returns a FactBase with all three sources populated

- [ ] **Step 3: Implement fact_builder.cpp**

`MapDetection`: Populate FactBase keys from DetectionResult fields.
`ResolveGraphPaths`: For each path, call `kg_->ResolvePath()`, flatten, `fb.Set()`.
`RunVectorRetrieval`: Call `vp_->TopK()`, expand results into `retrieval.top1.score`, etc.

- [ ] **Step 4: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "FactBuilderTest" --output-on-failure
```

Expected: All FactBuilderTest tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/sai/rule/fact_builder.h src/rule/fact_builder.cpp tests/rule/fact_builder_test.cpp
git commit -m "feat(rule): ✨ FactBuilder（DetectionResult 映射 + 图路径物化 + 向量检索注入）"
```

---

### Task 9: RuleEngine + TraceStep + Rule parsing from YAML

**Files:**
- Create: `include/sai/rule/rule_engine.h`
- Create: `src/rule/rule_engine.cpp`
- Create: `src/rule/conflict_resolver.cpp` (internal)
- Create: `tests/rule/rule_engine_test.cpp`
- Create: `tests/rule/conflict_resolver_test.cpp`

**Interfaces:**
- Produces: `TraceStep`, `Rule`, `RuleAction`, `ResolvedRule`, `OverlapWarning`, `RuleEngine`
- Consumes: AST (Task 7), FactBase (Task 6), Lexer (Task 4), Parser (Task 5), yaml-cpp

- [ ] **Step 1: Write rule_engine.h**

```cpp
// include/sai/rule/rule_engine.h
#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>
#include "sai/core/error.h"
#include "sai/rule/expression.h"
#include "sai/rule/fact_base.h"

namespace sai::rule {

struct RuleAction {
    std::string label;
    double base_severity{0.0};
    std::string recommendation;
};

struct TraceStep {
    std::string id;
    enum class Level { Expression, Rule, TreeBranch, Scoring };
    Level level;
    std::string description;
    std::string source_location;
    std::chrono::microseconds timestamp;
    std::optional<std::string> parent_id;
};

struct Rule {
    std::string name;
    uint32_t priority{0};
    std::string rule_set;
    std::vector<std::string> overrides;
    std::vector<std::string> overridden_by;
    std::unique_ptr<IExpression> condition;
    RuleAction action;
};

struct ResolvedRule {
    std::string name;
    bool matched;
    RuleAction action;
    std::vector<TraceStep> eval_trace;
};

struct OverlapWarning {
    std::string rule_a;
    std::string rule_b;
    std::vector<std::string> common_fields;
};

class RuleEngine {
public:
    auto LoadFromYAML(std::filesystem::path) -> Result<void>;
    auto EnableHotReload(std::filesystem::path, std::stop_token) -> Result<void>;
    auto EvaluateAll(FactBase&) -> Result<std::vector<ResolvedRule>>;
    auto ResolveConflicts(const std::vector<ResolvedRule>&) -> std::vector<ResolvedRule>;
    auto DetectOverlaps() const -> std::vector<OverlapWarning>;

private:
    auto EvaluateRuleSet(FactBase&, const std::vector<Rule>&) -> std::vector<ResolvedRule>;
    std::map<std::string, std::vector<Rule>> rule_sets_;
    std::filesystem::path current_path_;
};

}  // namespace sai::rule
```

- [ ] **Step 2: Write rule_engine_test.cpp**

Key tests:
- `LoadFromYAML`: parse a temp YAML file with 2 rule_sets, verify rules are loaded
- `EvaluateAll`: with FactBase populated, verify rule matching
- `EvaluateAll_ParallelRuleSets`: multiple rule_sets evaluate concurrently
- `EvaluateAll_SingleRuleFailureSkips`: one rule with invalid expression doesn't crash the engine
- `EnableHotReload`: reload YAML, verify old rules replaced (Linux-only, disabled on macOS)
- `DetectOverlaps`: two rules with overlapping field refs, no overrides → warning generated

- [ ] **Step 3: Write conflict_resolver_test.cpp**

```cpp
// tests/rule/conflict_resolver_test.cpp
#include <sai/rule/rule_engine.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(ConflictResolverTest, HigherPriorityWinsWithoutOverrides) {
    // Two rules matched, A has priority 200, B has priority 50, no overrides
    // → both survive (no explicit overrides = no conflict resolution by priority alone)
    // Actually per spec: overrides are explicit, priority is only tiebreaker.
    // Both survive unless explicit override.
}

TEST(ConflictResolverTest, ExplicitOverrideEliminatesTarget) {
    // A.overrides = [B], A matched, B matched → B is eliminated
}

TEST(ConflictResolverTest, OverriddenBySymmetrical) {
    // A.overrides = [B], B.overridden_by = [A] → B eliminated
}

TEST(ConflictResolverTest, CyclicOverrideDetected) {
    // A.overrides=[B], B.overrides=[C], C.overrides=[A] → Rule_CyclicOverride error
}

TEST(ConflictResolverTest, NonMatchedRuleNotOverridden) {
    // A.overrides=[B], A matched, B NOT matched → B stays unmatched
}

}  // namespace
}  // namespace sai::rule
```

- [ ] **Step 4: Implement rule_engine.cpp + conflict_resolver.cpp**

`LoadFromYAML`: uses yaml-cpp to parse rule YAML, calls Lexer+Parser for each rule's condition string.
`EvaluateAll`: iterates rule_sets, for each rule calls `condition->Evaluate(facts)`, records TraceStep for each evaluation.
`EvaluateRuleSet`: single rule_set evaluation (used for parallel dispatch).
`ResolveConflicts`: builds overrides DAG, topological elimination, cyclic detection.
`DetectOverlaps`: O(n²) field ref intersection check within each rule_set.
`EnableHotReload`: inotify on Linux, no-op on macOS (commented as placeholder).

Error recovery: `EvaluateAll` catches `Rule_TypeMismatch` per rule → logs, marks `matched=false`, continues.

- [ ] **Step 5: Build and run tests**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "RuleEngineTest|ConflictResolverTest" --output-on-failure
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/sai/rule/rule_engine.h src/rule/rule_engine.cpp src/rule/conflict_resolver.cpp tests/rule/rule_engine_test.cpp tests/rule/conflict_resolver_test.cpp
git commit -m "feat(rule): ✨ RuleEngine（YAML 加载 + 并行求值 + 冲突消解 + 热重载）"
```

---

### Task 10: 5.1 Integration test (Rule Engine end-to-end)

**Files:**
- Create: `tests/rule/integration_test.cpp`

**Interfaces:**
- Consumes: All 5.1 interfaces (Value, AST, Lexer, Parser, FactBase, FactBuilder, RuleEngine)

- [ ] **Step 1: Write integration test**

```cpp
// tests/rule/integration_test.cpp
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/rule/ast_nodes.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(RuleIntegrationTest, EndToEndRuleEvaluation) {
    // 1. Build FactBase with simulated detection data
    FactBase fb;
    fb.Set("defect.type", Value::Of(std::string("scratch")), FactSource{FactSourceKind::Direct});
    fb.Set("defect.area_mm2", Value::Of(12.3), FactSource{FactSourceKind::Direct});
    fb.Set("defect.position", Value::Of(std::string("seat_center")), FactSource{FactSourceKind::Direct});
    fb.Set("defect.score", Value::Of(0.92), FactSource{FactSourceKind::Direct});
    fb.Set("defect.confidence", Value::Of(0.85), FactSource{FactSourceKind::Direct});
    fb.Set("material.supplier.batch.reject_rate", Value::Of(0.032), FactSource{FactSourceKind::GraphPath});

    // 2. Parse rules from YAML
    RuleEngine engine;
    auto load = engine.LoadFromYAML("tests/rule/test_rules.yaml");
    ASSERT_TRUE(load.has_value());

    // 3. Evaluate
    auto results = engine.EvaluateAll(fb);
    ASSERT_TRUE(results.has_value());
    
    // 4. Resolve conflicts
    auto resolved = engine.ResolveConflicts(*results);
    
    // 5. Verify scratch_major matched, scratch_minor overridden
    bool found_major = false, found_minor_overridden = false;
    for (auto& r : resolved) {
        if (r.name == "scratch_major") found_major = r.matched;
        // scratch_minor should be in overridden list (not in resolved triggered)
    }
    EXPECT_TRUE(found_major);
}

TEST(RuleIntegrationTest, ErrorRecoverySkipsBadRule) {
    FactBase fb;
    // Rule with type mismatch: "string" > 5
    // Engine should skip this rule, not crash

    RuleEngine engine;
    // Construct a rule programmatically with bad expression
    // ... evaluate, verify engine returns partial results
}

}  // namespace
}  // namespace sai::rule
```

- [ ] **Step 2: Create test fixture YAML**

```yaml
# tests/rule/test_rules.yaml
rule_sets:
  test_defects:
    rules:
      - name: scratch_major
        priority: 100
        overrides: [scratch_minor]
        condition: |
          defect.type == "scratch" AND
          defect.area_mm2 > 10 AND
          defect.position IN ["seat_center", "side_bolster"]
        action:
          label: "scratch_major"
          base_severity: 0.8
          recommendation: "严重划痕，建议立即返工"
      
      - name: scratch_minor
        priority: 50
        overridden_by: [scratch_major]
        condition: |
          defect.type == "scratch" AND
          defect.area_mm2 > 2
        action:
          label: "scratch_minor"
          base_severity: 0.3
          recommendation: "轻微划痕，标记跟踪"
```

- [ ] **Step 3: Build and run integration test**

```bash
cmake --build --preset default --target sai_rule_tests && ctest --preset default -R "RuleIntegrationTest" --output-on-failure
```

Expected: Integration tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/rule/integration_test.cpp tests/rule/test_rules.yaml
git commit -m "test(rule): ✅ 5.1 集成测试（规则加载 → 求值 → 冲突消解）"
```

- [ ] **Step 5: Full 5.1 test run**

```bash
ctest --preset default -R "sai_rule_tests" --output-on-failure
```

Expected: All sai_rule_tests PASS.

---

### Task 11: DecisionTree

**Files:**
- Create: `include/sai/reasoner/decision_tree.h`
- Create: `src/reasoner/decision_tree.cpp`
- Create: `tests/reasoner/decision_tree_test.cpp`

**Interfaces:**
- Produces: `ScoreFormula`, `IDecisionNode`, `BranchNode`, `LeafNode`, `DecisionTree`
- Consumes: FactBase (5.1), yaml-cpp

- [ ] **Step 1: Write decision_tree.h**

```cpp
// include/sai/reasoner/decision_tree.h
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "sai/core/object.h"
#include "sai/core/error.h"

namespace sai::reasoner {

struct ScoreFormula {
    std::vector<double> weights;
    std::vector<std::string> features;
    double threshold{0.0};
};

class IDecisionNode : public core::Object {
public:
    enum class Kind { Branch, Leaf };
    virtual auto GetKind() const noexcept -> Kind = 0;
};

class BranchNode final : public IDecisionNode {
public:
    auto GetKind() const noexcept -> Kind override { return Kind::Branch; }
    auto FieldName() const -> std::string_view;
    auto Branches() const -> const std::map<std::string, std::unique_ptr<IDecisionNode>>&;
    auto DefaultBranch() const -> const IDecisionNode*;

    // Builder API (used by DecisionTree::LoadFromYAML)
    auto SetField(std::string field) -> void;
    auto AddBranch(std::string match_value, std::unique_ptr<IDecisionNode>) -> void;
    auto SetDefault(std::unique_ptr<IDecisionNode>) -> void;

private:
    std::string field_;
    std::map<std::string, std::unique_ptr<IDecisionNode>> branches_;
    std::unique_ptr<IDecisionNode> default_;
};

class LeafNode final : public IDecisionNode {
public:
    auto GetKind() const noexcept -> Kind override { return Kind::Leaf; }
    auto Formulas() const -> const std::vector<ScoreFormula>&;
    auto Label() const -> std::string_view;
    auto Recommendation() const -> std::string_view;

    auto SetLabel(std::string) -> void;
    auto SetRecommendation(std::string) -> void;
    auto AddFormula(ScoreFormula) -> void;

private:
    std::vector<ScoreFormula> formulas_;
    std::string label_;
    std::string recommendation_;
};

class DecisionTree {
public:
    static auto LoadFromYAML(std::filesystem::path) -> Result<std::unique_ptr<DecisionTree>>;
    auto Root() const -> const IDecisionNode&;

    explicit DecisionTree(std::unique_ptr<IDecisionNode> root);

private:
    static auto ParseNode(const YAML::Node& yaml) -> Result<std::unique_ptr<IDecisionNode>>;
    std::unique_ptr<IDecisionNode> root_;
};

}  // namespace sai::reasoner
```

- [ ] **Step 2: Write decision_tree_test.cpp**

Key tests:
- `LoadFromYAML_BasicTree`: minimal tree with one branch → leaf → verify structure
- `LoadFromYAML_MultiLevel`: scratch → area >10 / default
- `LoadFromYAML_MissingDefault`: branch without default → `Reasoner_InvalidTree`
- `LoadFromYAML_LeafWithFormulas`: verify ScoreFormula parsing
- `LoadFromYAML_FileNotFound`: `Reasoner_TreeLoadFailed`
- `LoadFromYAML_NumericBranchKeys`: `">10"`, `"<5"`, `">=3"` as branch keys

- [ ] **Step 3: Implement decision_tree.cpp**

Recursive YAML parsing:
- `type: branch` → construct BranchNode, recurse into branches + default
- `type: leaf` → construct LeafNode, parse formulas array
- Validation: branch MUST have default child → else `Reasoner_InvalidTree`

- [ ] **Step 4: Build and run tests**

```bash
cmake --build --preset default --target sai_reasoner_tests && ctest --preset default -R "DecisionTreeTest" --output-on-failure
```

Expected: All DecisionTreeTest tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/sai/reasoner/decision_tree.h src/reasoner/decision_tree.cpp tests/reasoner/decision_tree_test.cpp
git commit -m "feat(reasoner): ✨ DecisionTree（YAML → BranchNode/LeafNode + 校验）"
```

---

### Task 12: ScoreCalculator + TraceRecorder + EvidenceCollector

**Files:**
- Create: `src/reasoner/score_calculator.cpp` (+ internal header)
- Create: `src/reasoner/trace_recorder.cpp` (+ internal header)
- Create: `src/reasoner/evidence_collector.cpp` (+ internal header)
- Create: `tests/reasoner/score_calculator_test.cpp`

**Interfaces:**
- Produces: `ScoreCalculator`, `TraceRecorder`, `EvidenceCollector` (internal implementation helpers for DefaultReasoner)
- Consumes: ScoreFormula, FactBase, TraceStep (5.1)

- [ ] **Step 1: Implement score_calculator.cpp**

```cpp
// src/reasoner/score_calculator.cpp
#include "score_calculator.h"
#include <cmath>
#include "sai/rule/fact_base.h"

namespace sai::reasoner {

auto ScoreCalculator::Compute(const ScoreFormula& formula, const rule::FactBase& facts) -> double {
    double sum = 0.0;
    for (size_t i = 0; i < formula.features.size(); ++i) {
        auto v = facts.Get(formula.features[i]);
        if (v.has_value()) {
            auto d = v->AsDouble();
            if (d.has_value()) sum += formula.weights[i] * (*d);
        }
        // Missing feature → contributes 0.0 (partial scoring)
    }
    double raw = sum - formula.threshold;
    return Sigmoid(raw);
}

auto ScoreCalculator::ComputeMax(const std::vector<ScoreFormula>& formulas, const rule::FactBase& facts) -> double {
    double best = 0.0;
    for (auto& f : formulas) {
        double s = Compute(f, facts);
        if (s > best) best = s;
    }
    return best;
}

double ScoreCalculator::Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace sai::reasoner
```

- [ ] **Step 2: Write score_calculator_test.cpp**

Tests:
- `Sigmoid_AtZero`: sigmoid(0) ≈ 0.5
- `Sigmoid_LargePositive`: sigmoid(10) ≈ 1.0
- `Sigmoid_LargeNegative`: sigmoid(-10) ≈ 0.0
- `Compute_SingleFormula`: known weights + features → verify output
- `Compute_MissingFeatureContributesZero`: feature not in FactBase → contributes 0
- `ComputeMax_ReturnsBestFormula`: two formulas, verify max is selected

- [ ] **Step 3: Implement trace_recorder.cpp**

`TraceRecorder` accumulates `TraceStep` objects during tree traversal:
```cpp
auto RecordExpression(std::string desc, std::string source) -> std::string;  // returns step ID
auto RecordRule(std::string rule_name, bool matched) -> std::string;
auto RecordTreeBranch(std::string field, std::string value, std::string branch) -> std::string;
auto RecordScoring(std::string formula_desc, double score) -> std::string;
auto AllSteps() const -> std::vector<TraceStep>;
```

- [ ] **Step 4: Implement evidence_collector.cpp**

`EvidenceCollector::Pack(facts, trace, rules)` → assembles `EvidenceItem[]` from:
- `FactBase::AllEntries()` + `FactBase::AllSources()`
- Links each item to a TraceStep ID via parent key

- [ ] **Step 5: Build and run tests**

```bash
cmake --build --preset default --target sai_reasoner_tests && ctest --preset default -R "ScoreCalculatorTest" --output-on-failure
```

Expected: All ScoreCalculatorTest tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reasoner/score_calculator.h src/reasoner/score_calculator.cpp src/reasoner/trace_recorder.h src/reasoner/trace_recorder.cpp src/reasoner/evidence_collector.h src/reasoner/evidence_collector.cpp tests/reasoner/score_calculator_test.cpp
git commit -m "feat(reasoner): ✨ ScoreCalculator + TraceRecorder + EvidenceCollector"
```

---

### Task 13: IReasoner + DefaultReasoner + TreeWalker

**Files:**
- Create: `include/sai/reasoner/reasoner.h`
- Create: `src/reasoner/reasoner.cpp`
- Create: `src/reasoner/tree_walker.cpp` (+ internal header)
- Create: `tests/reasoner/reasoner_test.cpp`

**Interfaces:**
- Produces: `IReasoner`, `DefaultReasoner`, `ReasoningResult`, `EvidenceItem`
- Consumes: DecisionTree (Task 11), ScoreCalculator/TraceRecorder/EvidenceCollector (Task 12), FactBase/ResolvedRule (5.1)

- [ ] **Step 1: Write reasoner.h**

```cpp
// include/sai/reasoner/reasoner.h
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "sai/core/error.h"
#include "sai/core/object.h"
#include "sai/core/service.h"
#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"
#include "sai/reasoner/decision_tree.h"

namespace sai::reasoner {

struct EvidenceItem {
    std::string key;
    rule::Value value;
    rule::FactSource source;
};

struct ReasoningResult {
    std::string verdict;
    double severity{0.0};
    std::string recommendation;
    double confidence{0.0};
    std::vector<rule::TraceStep> trace;
    std::vector<EvidenceItem> evidence;
    std::vector<std::string> triggered_rules;
    std::vector<std::string> overridden_rules;
};

class IReasoner : public core::IService {
public:
    virtual auto Reason(
        const rule::FactBase& facts,
        const std::vector<rule::ResolvedRule>& rules
    ) -> Result<ReasoningResult> = 0;
};

class DefaultReasoner final : public IReasoner {
public:
    explicit DefaultReasoner(std::unique_ptr<DecisionTree> tree);
    auto Reason(const rule::FactBase&, const std::vector<rule::ResolvedRule>&)
        -> Result<ReasoningResult> override;

private:
    std::unique_ptr<DecisionTree> tree_;
};

}  // namespace sai::reasoner
```

- [ ] **Step 2: Implement tree_walker.cpp**

```cpp
// src/reasoner/tree_walker.cpp
#include "tree_walker.h"
#include "score_calculator.h"
#include "trace_recorder.h"

namespace sai::reasoner {

struct WalkResult {
    std::string label;
    double score;
    std::string recommendation;
};

auto TreeWalker::Walk(const IDecisionNode& node, const rule::FactBase& facts,
                       TraceRecorder& tracer) -> WalkResult {
    if (node.GetKind() == IDecisionNode::Kind::Leaf) {
        auto& leaf = static_cast<const LeafNode&>(node);
        double score = ScoreCalculator::ComputeMax(leaf.Formulas(), facts);
        tracer.RecordScoring(
            std::string("Leaf: ") + std::string(leaf.Label()) + " score=" + std::to_string(score), score);
        return {std::string(leaf.Label()), score, std::string(leaf.Recommendation())};
    }

    auto& branch = static_cast<const BranchNode&>(node);
    auto field_val = facts.Get(branch.FieldName());
    
    if (field_val.has_value()) {
        // Try string match
        if (auto s = field_val->AsString()) {
            auto it = branch.Branches().find(std::string(*s));
            if (it != branch.Branches().end()) {
                tracer.RecordTreeBranch(std::string(branch.FieldName()), std::string(*s), std::string(*s));
                return Walk(*it->second, facts, tracer);
            }
        }
        // Try numeric match (">10", "<5", etc.)
        if (auto d = field_val->AsDouble()) {
            for (auto& [key, child] : branch.Branches()) {
                if (MatchNumeric(key, *d)) {
                    tracer.RecordTreeBranch(std::string(branch.FieldName()), std::to_string(*d), key);
                    return Walk(*child, facts, tracer);
                }
            }
        }
    }
    
    // Fallback to default
    tracer.RecordTreeBranch(std::string(branch.FieldName()),
        field_val.has_value() ? "value" : "missing", "default");
    if (!branch.DefaultBranch()) {
        // Should never happen — validated at load time
        return {"UNCERTAIN", 0.0, "No matching branch and no default"};
    }
    return Walk(*branch.DefaultBranch(), facts, tracer);
}

bool TreeWalker::MatchNumeric(const std::string& key, double value) {
    if (key.empty()) return false;
    char op = key[0];
    double threshold = std::stod(key.substr(1));
    switch (op) {
    case '>': return key.size() > 1 && key[1] == '=' ? value >= threshold : value > threshold;
    case '<': return key.size() > 1 && key[1] == '=' ? value <= threshold : value < threshold;
    default: return false;
    }
}

}  // namespace sai::reasoner
```

- [ ] **Step 3: Implement reasoner.cpp (DefaultReasoner)**

```cpp
// src/reasoner/reasoner.cpp
#include "sai/reasoner/reasoner.h"
#include "tree_walker.h"
#include "trace_recorder.h"
#include "evidence_collector.h"

namespace sai::reasoner {

DefaultReasoner::DefaultReasoner(std::unique_ptr<DecisionTree> tree)
    : tree_(std::move(tree)) {}

auto DefaultReasoner::Reason(
    const rule::FactBase& facts,
    const std::vector<rule::ResolvedRule>& rules
) -> Result<ReasoningResult> {
    TraceRecorder tracer;
    EvidenceCollector evidence;
    
    // Inject rule results into trace
    std::vector<std::string> triggered, overridden;
    for (auto& r : rules) {
        if (r.matched) {
            triggered.push_back(r.name);
            tracer.RecordRule(r.name, true);
            // Merge rule-level eval_trace
            for (auto& step : r.eval_trace) {
                // Add to trace with parent reference
            }
        } else {
            overridden.push_back(r.name);
        }
    }
    
    // Walk the decision tree
    auto walk_result = TreeWalker::Walk(tree_->Root(), facts, tracer);
    
    // Determine verdict
    std::string verdict;
    if (walk_result.label == "NG" || walk_result.label == "WARN" || walk_result.label == "OK" ||
        walk_result.label == "UNCERTAIN") {
        verdict = walk_result.label;
    } else if (walk_result.score > 0.7) {
        verdict = "NG";
    } else if (walk_result.score > 0.3) {
        verdict = "WARN";
    } else {
        verdict = "OK";
    }
    
    // Assemble result
    ReasoningResult result;
    result.verdict = verdict;
    result.severity = walk_result.score;
    result.recommendation = walk_result.recommendation;
    result.confidence = walk_result.score;  // reuse leaf score
    result.trace = tracer.AllSteps();
    result.evidence = evidence.Pack(facts, tracer.AllSteps(), rules);
    result.triggered_rules = std::move(triggered);
    result.overridden_rules = std::move(overridden);
    
    return result;
}

}  // namespace sai::reasoner
```

- [ ] **Step 4: Write reasoner_test.cpp**

Key tests:
- `Reason_SimpleTree`: FactBase with scratch defect → decision tree → NG verdict
- `Reason_DefaultBranch`: defect type "unknown" → default leaf → UNCERTAIN
- `Reason_TraceContainsAllLevels`: verify trace has Expression, Rule, TreeBranch, Scoring steps
- `Reason_EvidenceContainsFactSources`: verify evidence items have FactSource metadata
- `Reason_MissingFieldGoesToDefault`: branch field not in FactBase → default branch

- [ ] **Step 5: Build and run tests**

```bash
cmake --build --preset default --target sai_reasoner_tests && ctest --preset default -R "ReasonerTest" --output-on-failure
```

Expected: All ReasonerTest tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/sai/reasoner/reasoner.h src/reasoner/reasoner.cpp src/reasoner/tree_walker.h src/reasoner/tree_walker.cpp tests/reasoner/reasoner_test.cpp
git commit -m "feat(reasoner): ✨ IReasoner + DefaultReasoner（决策树遍历 + 评分 + Trace + Evidence）"
```

---

### Task 14: M5 End-to-end integration test + Contract update

**Files:**
- Create: `tests/reasoner/integration_test.cpp`
- Create: YAML fixtures for rules + decision tree
- Modify: `tests/integration/CMakeLists.txt` (add M5 integration test)
- Modify: `docs/surface-ai/glossary-and-contracts.md` (append M5 contract increment)

**Interfaces:**
- Consumes: All M5 interfaces, M3 DetectionResult, M4 KnowledgeGraph

- [ ] **Step 1: Write M5 pipeline integration test**

```cpp
// tests/reasoner/integration_test.cpp
// End-to-end: DetectionResult → FactBase → RuleEngine → Reasoner → ReasoningResult

TEST(M5IntegrationTest, FullPipelineScratchDefect) {
    // 1. Simulate DetectionResult (hand-crafted, no real inference)
    // 2. FactBuilder builds FactBase with detection data + mocked graph paths
    // 3. RuleEngine loads test_rules.yaml, evaluates against FactBase
    // 4. Reasoner loads test_decision_tree.yaml
    // 5. Reasoner::Reason produces ReasoningResult
    // 6. Verify: verdict="NG", severity > 0.5, trace has 4 levels, evidence has FactSources
}
```

- [ ] **Step 2: Add M5 integration test to tests/integration/CMakeLists.txt**

```cmake
# M5: Rule → Reasoner 端到端集成测试
add_executable(sai_integration_rule_reasoner_pipeline_test
    rule_reasoner_pipeline_test.cpp
)
target_include_directories(sai_integration_rule_reasoner_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_rule_reasoner_pipeline_test PRIVATE
    sai::reasoner
    sai::rule
    sai::knowledge
    sai::retrieval
    sai::detection
    GTest::gtest_main
)
gtest_discover_tests(sai_integration_rule_reasoner_pipeline_test)
```

- [ ] **Step 3: Update glossary-and-contracts.md**

Append M5 concepts and interface signatures from spec §19.1 and §19.2.

- [ ] **Step 4: Build and run all tests**

```bash
cmake --build --preset default && ctest --preset default --output-on-failure
```

Expected: All existing tests pass + all new M5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/reasoner/integration_test.cpp tests/integration/CMakeLists.txt tests/integration/rule_reasoner_pipeline_test.cpp docs/surface-ai/glossary-and-contracts.md
git commit -m "test(reasoner): ✅ M5 端到端集成测试 + 契约表更新"
```

---

### Task 15: Final review — run all tests, verify verification point

**Files:**
- None (verification only)

- [ ] **Step 1: Full test suite**

```bash
cmake --build --preset default && ctest --preset default --output-on-failure
```

Expected: All tests pass, no regressions.

- [ ] **Step 2: Verify M5 verification point (spec §18)**

Run the integration test that exercises the verification scenario:
1. Car seat surface detected with scratch (area 12.3mm², position seat_center)
2. FactBase populated from simulated DetectionResult + graph path materialization
3. RuleEngine → scratch_major matched, scratch_minor overridden
4. DefaultReasoner → decision tree scratch branch → area > 10 branch → Leaf NG
5. ReasoningResult:
   - verdict = "NG"
   - trace[] contains Expression, Rule, TreeBranch, Scoring levels
   - evidence[] contains defect fields + graph path SQL + source metadata

- [ ] **Step 3: Manual trace inspection**

Run the integration test with verbose output, inspect the trace JSON:
```bash
cd build/default && ./tests/reasoner/sai_reasoner_tests --gtest_filter="*FullPipeline*" --gtest_print_time=1
```

Confirm trace output shows:
- Expression level: "defect.area_mm2 = 12.3"
- Rule level: "scratch_major: matched=true"
- TreeBranch level: "branch on defect.type='scratch' → scratch branch"
- Scoring level: "sigmoid(0.5*0.92 + 0.3*0.85 + 0.2*0.032 - 0.5) = ..."

- [ ] **Step 4: Commit final review notes**

```bash
git add .superpowers/sdd/
git commit -m "chore(sdd): 🔧 M5 最终回顾 — 验证点通过，全部测试通过"
```
