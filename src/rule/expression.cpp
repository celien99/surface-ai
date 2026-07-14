#include "sai/rule/ast_nodes.h"

namespace sai::rule {

// ---------------------------------------------------------------------------
// LiteralExpr
// ---------------------------------------------------------------------------
LiteralExpr::LiteralExpr(Value val)
    : value_(std::move(val))
    , source_text_()
{
    auto d = value_.AsDouble();
    if (d.has_value()) {
        source_text_ = std::to_string(*d);
        return;
    }
    auto s = value_.AsString();
    if (s.has_value()) {
        source_text_ = std::string(*s);
        return;
    }
    auto b = value_.AsBool();
    if (b.has_value()) {
        source_text_ = (*b) ? "true" : "false";
        return;
    }
    source_text_ = "null";
}

auto LiteralExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto LiteralExpr::SourceText() const -> std::string_view {
    return source_text_;
}

auto LiteralExpr::GetValue() const -> const Value& {
    return value_;
}

// ---------------------------------------------------------------------------
// FieldRefExpr
// ---------------------------------------------------------------------------
FieldRefExpr::FieldRefExpr(std::string key)
    : key_(std::move(key))
{
}

auto FieldRefExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto FieldRefExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return {key_};
}

auto FieldRefExpr::SourceText() const -> std::string_view {
    return key_;
}

// ---------------------------------------------------------------------------
// BinaryExpr
// ---------------------------------------------------------------------------
BinaryExpr::BinaryExpr(BinaryOp op, std::unique_ptr<IExpression> lhs,
                       std::unique_ptr<IExpression> rhs, std::string source)
    : op_(op)
    , lhs_(std::move(lhs))
    , rhs_(std::move(rhs))
    , source_text_(std::move(source))
{
}

auto BinaryExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto BinaryExpr::CollectFieldRefs() const -> std::vector<std::string> {
    std::vector<std::string> refs;
    auto l = lhs_->CollectFieldRefs();
    auto r = rhs_->CollectFieldRefs();
    refs.reserve(l.size() + r.size());
    refs.insert(refs.end(), l.begin(), l.end());
    refs.insert(refs.end(), r.begin(), r.end());
    return refs;
}

auto BinaryExpr::SourceText() const -> std::string_view {
    return source_text_;
}

auto BinaryExpr::GetOp() const -> BinaryOp {
    return op_;
}

auto BinaryExpr::GetLhs() const -> const IExpression& {
    return *lhs_;
}

auto BinaryExpr::GetRhs() const -> const IExpression& {
    return *rhs_;
}

// ---------------------------------------------------------------------------
// UnaryExpr
// ---------------------------------------------------------------------------
UnaryExpr::UnaryExpr(UnaryOp op, std::unique_ptr<IExpression> operand, std::string source)
    : op_(op)
    , operand_(std::move(operand))
    , source_text_(std::move(source))
{
}

auto UnaryExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto UnaryExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return operand_->CollectFieldRefs();
}

auto UnaryExpr::SourceText() const -> std::string_view {
    return source_text_;
}

// ---------------------------------------------------------------------------
// FunctionExpr
// ---------------------------------------------------------------------------
FunctionExpr::FunctionExpr(BuiltinFn fn, std::vector<std::unique_ptr<IExpression>> args,
                           std::string source)
    : fn_(fn)
    , args_(std::move(args))
    , source_text_(std::move(source))
{
}

auto FunctionExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto FunctionExpr::CollectFieldRefs() const -> std::vector<std::string> {
    std::vector<std::string> refs;
    for (const auto& arg : args_) {
        auto sub = arg->CollectFieldRefs();
        refs.insert(refs.end(), sub.begin(), sub.end());
    }
    return refs;
}

auto FunctionExpr::SourceText() const -> std::string_view {
    return source_text_;
}

// ---------------------------------------------------------------------------
// PathExpr
// ---------------------------------------------------------------------------
PathExpr::PathExpr(std::string path)
    : path_(std::move(path))
{
    // Flatten arrow notation "->" to dots
    flattened_key_.reserve(path_.size());
    for (size_t i = 0; i < path_.size(); ++i) {
        if (path_[i] == '-' && i + 1 < path_.size() && path_[i + 1] == '>') {
            flattened_key_.push_back('.');
            ++i;  // skip '>'
        } else {
            flattened_key_.push_back(path_[i]);
        }
    }
}

auto PathExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    return Result<Value>(Value::Null());  // stub — Task 6
}

auto PathExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return {path_};
}

auto PathExpr::SourceText() const -> std::string_view {
    return path_;
}

auto PathExpr::GetPath() const -> std::string_view {
    return path_;
}

auto PathExpr::GetFlattenedKey() const -> std::string_view {
    return flattened_key_;
}

}  // namespace sai::rule
