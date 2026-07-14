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

class IExpression : public sai::Object {
public:
    virtual auto GetKind() const noexcept -> ExprKind = 0;
    virtual auto Evaluate(FactBase& ctx) const -> Result<Value> = 0;
    virtual auto CollectFieldRefs() const -> std::vector<std::string> = 0;
    virtual auto SourceText() const -> std::string_view = 0;
};

}  // namespace sai::rule
