// condition_builder.h — Build IExpression AST from structured YAML (internal)
#pragma once

#include <memory>
#include <yaml-cpp/yaml.h>
#include <sai/core/error.h>

namespace sai::rule {
class IExpression;

// Build an IExpression tree from a structured YAML condition node.
// Returns nullptr if the node is not defined (empty condition = always match).
//
// Supported formats:
//   {field: "a.b", op: "gt", value: 0.8}
//   {and: [{field: a, op: gt, value: 0.5}, {field: b, op: lt, value: 100}]}
//   {or: [...]}
//   {not: {field: a, op: eq, value: 0}}
//   {path: "material->supplier->batch"}
//   {value: 42}
auto BuildExpressionFromYAML(const YAML::Node& node)
    -> Result<std::unique_ptr<IExpression>>;

}  // namespace sai::rule
