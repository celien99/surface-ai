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
