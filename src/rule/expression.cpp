#include "sai/rule/ast_nodes.h"
#include "sai/rule/fact_base.h"

#include <functional>   // std::hash
#include <string>
#include <string_view>

namespace sai::rule {
namespace {

// ---------------------------------------------------------------------------
// Memoization helper
//   Cache key = "__memo__" + std::hash(source_text)
// ---------------------------------------------------------------------------
auto MemoKey(std::string_view src) -> std::string {
    return "__memo__" + std::to_string(std::hash<std::string_view>{}(src));
}

template <typename Fn>
auto EvalWithMemo(FactBase& ctx, std::string_view src, Fn&& compute)
    -> Result<Value> {
    auto key = MemoKey(src);
    if (auto cached = ctx.Get(key)) {
        return std::move(*cached);
    }
    auto result = compute();
    if (result.has_value()) {
        FactSource memo_src;
        memo_src.kind   = FactSourceKind::Computed;
        memo_src.description = "memoized:" + std::string(src);
        ctx.Set(key, *result, std::move(memo_src));
    }
    return result;
}

}  // namespace

// ===========================================================================
// LiteralExpr
// ===========================================================================

LiteralExpr::LiteralExpr(Value val)
    : value_(std::move(val))
    , source_text_()
{
    if (auto d = value_.AsDouble()) {
        source_text_ = std::to_string(*d);
        return;
    }
    if (auto s = value_.AsString()) {
        source_text_ = std::string(*s);
        return;
    }
    if (auto b = value_.AsBool()) {
        source_text_ = (*b) ? "true" : "false";
        return;
    }
    source_text_ = "null";
}

auto LiteralExpr::Evaluate(FactBase& /*ctx*/) const -> Result<Value> {
    // Literal is trivially cheap — no memoization overhead needed.
    return value_;
}

auto LiteralExpr::SourceText() const -> std::string_view {
    return source_text_;
}

auto LiteralExpr::GetValue() const -> const Value& {
    return value_;
}

// ===========================================================================
// FieldRefExpr
// ===========================================================================

FieldRefExpr::FieldRefExpr(std::string key)
    : key_(std::move(key)) {}

auto FieldRefExpr::Evaluate(FactBase& ctx) const -> Result<Value> {
    return EvalWithMemo(ctx, key_, [this, &ctx]() -> Result<Value> {
        auto val = ctx.Get(key_);
        if (val.has_value()) {
            return std::move(*val);
        }
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_TypeMismatch,
            "field not found: " + key_,
            std::source_location::current()});
    });
}

auto FieldRefExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return {key_};
}

auto FieldRefExpr::SourceText() const -> std::string_view {
    return key_;
}

// ===========================================================================
// BinaryExpr
// ===========================================================================

BinaryExpr::BinaryExpr(BinaryOp op, std::unique_ptr<IExpression> lhs,
                       std::unique_ptr<IExpression> rhs, std::string source)
    : op_(op)
    , lhs_(std::move(lhs))
    , rhs_(std::move(rhs))
    , source_text_(std::move(source)) {}

auto BinaryExpr::Evaluate(FactBase& ctx) const -> Result<Value> {
    return EvalWithMemo(ctx, source_text_, [this, &ctx]() -> Result<Value> {
        // ---- Short-circuit: AND / OR ----
        if (op_ == BinaryOp::And || op_ == BinaryOp::Or) {
            auto lhs_res = lhs_->Evaluate(ctx);
            if (!lhs_res) {
                return tl::make_unexpected(lhs_res.error());
            }

            bool lhs_b = lhs_res->AsBool().value_or(false);

            // AND with lhs = false  → short-circuit to false
            if (op_ == BinaryOp::And && !lhs_b) {
                return Value::Of(false);
            }
            // OR  with lhs = true   → short-circuit to true
            if (op_ == BinaryOp::Or && lhs_b) {
                return Value::Of(true);
            }

            // No short-circuit: evaluate RHS
            auto rhs_res = rhs_->Evaluate(ctx);
            if (!rhs_res) {
                return tl::make_unexpected(rhs_res.error());
            }
            bool rhs_b = rhs_res->AsBool().value_or(false);
            return Value::Of(op_ == BinaryOp::And ? (lhs_b && rhs_b)
                                                  : (lhs_b || rhs_b));
        }

        // ---- Non-short-circuit operators: evaluate both sides ----
        auto lhs_res = lhs_->Evaluate(ctx);
        if (!lhs_res) {
            return tl::make_unexpected(lhs_res.error());
        }
        auto rhs_res = rhs_->Evaluate(ctx);
        if (!rhs_res) {
            return tl::make_unexpected(rhs_res.error());
        }

        const auto& lv = *lhs_res;
        const auto& rv = *rhs_res;

        switch (op_) {
            case BinaryOp::Add: return lv + rv;
            case BinaryOp::Sub: return lv - rv;
            case BinaryOp::Mul: return lv * rv;
            case BinaryOp::Div: return lv / rv;

            case BinaryOp::Eq:  return Value::Of(lv == rv);
            case BinaryOp::Neq: return Value::Of(!(lv == rv));

            case BinaryOp::Lt:  return Value::Of(lv < rv);
            case BinaryOp::Le:  return Value::Of(lv < rv || lv == rv);
            case BinaryOp::Gt:  return Value::Of(lv > rv);
            case BinaryOp::Ge:  return Value::Of(lv > rv || lv == rv);

            case BinaryOp::In: {
                auto* list = rv.AsList();
                if (!list) {
                    return Value::Of(false);
                }
                for (const auto& elem : *list) {
                    if (lv == elem) {
                        return Value::Of(true);
                    }
                }
                return Value::Of(false);
            }

            default:
                return Value::Null();
        }
    });
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

auto BinaryExpr::GetOp() const -> BinaryOp { return op_; }
auto BinaryExpr::GetLhs() const -> const IExpression& { return *lhs_; }
auto BinaryExpr::GetRhs() const -> const IExpression& { return *rhs_; }

// ===========================================================================
// UnaryExpr
// ===========================================================================

UnaryExpr::UnaryExpr(UnaryOp op, std::unique_ptr<IExpression> operand,
                     std::string source)
    : op_(op)
    , operand_(std::move(operand))
    , source_text_(std::move(source)) {}

auto UnaryExpr::Evaluate(FactBase& ctx) const -> Result<Value> {
    return EvalWithMemo(ctx, source_text_, [this, &ctx]() -> Result<Value> {
        auto operand_res = operand_->Evaluate(ctx);
        if (!operand_res) {
            return tl::make_unexpected(operand_res.error());
        }

        switch (op_) {
            case UnaryOp::Not: {
                auto b = operand_res->AsBool();
                if (!b) {
                    return Value::Null();
                }
                return Value::Of(!*b);
            }
            case UnaryOp::Neg: {
                if (auto d = operand_res->AsDouble()) {
                    return Value::Of(-*d);
                }
                if (auto b = operand_res->AsBool()) {
                    return Value::Of(-(*b ? 1.0 : 0.0));
                }
                return Value::Null();
            }
            default:
                return Value::Null();
        }
    });
}

auto UnaryExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return operand_->CollectFieldRefs();
}

auto UnaryExpr::SourceText() const -> std::string_view {
    return source_text_;
}

// ===========================================================================
// FunctionExpr
// ===========================================================================

FunctionExpr::FunctionExpr(BuiltinFn fn,
                           std::vector<std::unique_ptr<IExpression>> args,
                           std::string source)
    : fn_(fn)
    , args_(std::move(args))
    , source_text_(std::move(source)) {}

auto FunctionExpr::Evaluate(FactBase& ctx) const -> Result<Value> {
    return EvalWithMemo(ctx, source_text_, [this, &ctx]() -> Result<Value> {
        // Evaluate all arguments first
        std::vector<Value> values;
        values.reserve(args_.size());
        for (const auto& arg : args_) {
            auto v = arg->Evaluate(ctx);
            if (!v) {
                return tl::make_unexpected(v.error());
            }
            values.push_back(std::move(*v));
        }

        switch (fn_) {
            case BuiltinFn::Avg: {
                if (values.empty()) {
                    return Value::Null();
                }
                double sum = 0.0;
                for (const auto& v : values) {
                    auto n = v.AsDouble();
                    if (!n) {
                        return Value::Null();
                    }
                    sum += *n;
                }
                return Value::Of(sum / static_cast<double>(values.size()));
            }

            case BuiltinFn::Count:
                return Value::Of(static_cast<double>(values.size()));

            case BuiltinFn::Max: {
                if (values.empty()) {
                    return Value::Null();
                }
                auto opt = values[0].AsDouble();
                if (!opt) {
                    return Value::Null();
                }
                double m = *opt;
                for (size_t i = 1; i < values.size(); ++i) {
                    auto n = values[i].AsDouble();
                    if (!n) {
                        return Value::Null();
                    }
                    if (*n > m) {
                        m = *n;
                    }
                }
                return Value::Of(m);
            }

            case BuiltinFn::Min: {
                if (values.empty()) {
                    return Value::Null();
                }
                auto opt = values[0].AsDouble();
                if (!opt) {
                    return Value::Null();
                }
                double m = *opt;
                for (size_t i = 1; i < values.size(); ++i) {
                    auto n = values[i].AsDouble();
                    if (!n) {
                        return Value::Null();
                    }
                    if (*n < m) {
                        m = *n;
                    }
                }
                return Value::Of(m);
            }

            case BuiltinFn::Sum: {
                double sum = 0.0;
                for (const auto& v : values) {
                    auto n = v.AsDouble();
                    if (!n) {
                        return Value::Null();
                    }
                    sum += *n;
                }
                return Value::Of(sum);
            }

            case BuiltinFn::Len: {
                if (values.empty()) {
                    return Value::Of(0.0);
                }
                if (auto s = values[0].AsString()) {
                    return Value::Of(static_cast<double>(s->size()));
                }
                if (auto* list = values[0].AsList()) {
                    return Value::Of(static_cast<double>(list->size()));
                }
                return Value::Null();
            }
            default:
                return Value::Null();
        }
    });
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

// ===========================================================================
// PathExpr
// ===========================================================================

PathExpr::PathExpr(std::string path)
    : path_(std::move(path)) {
    // Flatten arrow notation "->" to dots at construction time.
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

auto PathExpr::Evaluate(FactBase& ctx) const -> Result<Value> {
    return EvalWithMemo(ctx, path_, [this, &ctx]() -> Result<Value> {
        // Try the registered path mapping first; fall back to
        // the built-in arrow→dot flattening.
        auto flat_key = ctx.ResolvePath(path_).value_or(flattened_key_);

        auto val = ctx.Get(flat_key);
        if (val.has_value()) {
            return std::move(*val);
        }
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_TypeMismatch,
            "field not found via path: " + std::string(flat_key),
            std::source_location::current()});
    });
}

auto PathExpr::CollectFieldRefs() const -> std::vector<std::string> {
    return {path_};
}

auto PathExpr::SourceText() const -> std::string_view {
    return path_;
}

auto PathExpr::GetPath() const -> std::string_view { return path_; }
auto PathExpr::GetFlattenedKey() const -> std::string_view {
    return flattened_key_;
}

}  // namespace sai::rule
