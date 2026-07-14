#pragma once

// -----------------------------------------------------------------------
// INTERNAL header — not part of the public sai::rule API.
// Declares the conflict resolution function used by RuleEngine.
// -----------------------------------------------------------------------

#include "sai/rule/rule_engine.h"

namespace sai::rule::detail {

// Build the override DAG from matched rules, detect cycles, and eliminate
// overridden rules by setting their `matched` flag to false.
//
// Pre-conditions:
//   - `evaluated` contains at least all rules from `rule_sets` that
//     were part of a prior EvaluateAll() call.
//   - Only rules with `matched == true` participate in the override graph.
//
// Post-conditions:
//   - On success: rules that are overridden by another matched rule have
//     `matched` set to false.
//   - On failure (Rule_CyclicOverride): `evaluated` is not modified.
//
// Cycle detection uses DFS colouring (White = unvisited, Gray = in current
// DFS path, Black = fully processed).
auto ResolveOverrideConflicts(
    const std::map<std::string, std::vector<Rule>>& rule_sets,
    std::vector<ResolvedRule>& evaluated) -> Result<void>;

}  // namespace sai::rule::detail
