#include "conflict_resolver.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "sai/core/error.h"

namespace sai::rule::detail {

// ===========================================================================
// ResolveOverrideConflicts
// ===========================================================================
//
// Algorithm:
//   1. Build a name → Rule lookup from rule_sets.
//   2. Collect which rules matched from the evaluated vector.
//   3. Build a directed graph where edge A → B means "A overrides B".
//      Edges come from both `overrides` (A.overrides=[…]) and
//      `overridden_by` (B.overridden_by=[A]).
//   4. DFS colouring cycle detection (White/Gray/Black).
//      Cycle → return Rule_CyclicOverride.
//   5. No cycles: for each matched rule, check if any other matched rule
//      has an edge pointing to it. If so, set matched = false (eliminated).
// ===========================================================================

auto ResolveOverrideConflicts(
    const std::map<std::string, std::vector<Rule>>& rule_sets,
    std::vector<ResolvedRule>& evaluated) -> Result<void> {
    // ---- 1. Build original rule lookup ----
    std::map<std::string, const Rule*> original;
    for (const auto& [set_name, rules] : rule_sets) {
        (void)set_name;
        for (const auto& r : rules) {
            original[r.name] = &r;
        }
    }

    // ---- 2. Identify matched rules ----
    std::map<std::string, std::size_t> matched_idx;  // name → index in evaluated
    for (std::size_t i = 0; i < evaluated.size(); ++i) {
        if (evaluated[i].matched) {
            matched_idx[evaluated[i].name] = i;
        }
    }

    // ---- 3. Build override graph among matched rules ----
    // graph[from] = [to, …]  where "from overrides to"
    std::map<std::string, std::vector<std::string>> graph;
    for (const auto& [name, idx] : matched_idx) {
        (void)idx;
        auto it = original.find(name);
        if (it == original.end()) continue;
        const auto* rule = it->second;

        // Direct overrides: A.overrides contains T → A overrides T
        for (const auto& target : rule->overrides) {
            if (matched_idx.contains(target)) {
                graph[name].push_back(target);
            }
        }

        // Reciprocal overridden_by: B.overridden_by contains M → M overrides B
        for (const auto& target_name : rule->overridden_by) {
            if (matched_idx.contains(target_name)) {
                graph[target_name].push_back(name);
            }
        }
    }

    // ---- 4. Cycle detection (DFS colouring) ----
    enum class Color { White, Gray, Black };
    std::map<std::string, Color> color;
    for (const auto& [name, _] : matched_idx) {
        color[name] = Color::White;
    }

    std::function<bool(const std::string&)> dfs_visit;
    dfs_visit = [&](const std::string& node) -> bool {
        color[node] = Color::Gray;
        for (const auto& neighbor : graph[node]) {
            if (color[neighbor] == Color::Gray) {
                return true;  // back edge → cycle
            }
            if (color[neighbor] == Color::White) {
                if (dfs_visit(neighbor)) return true;
            }
        }
        color[node] = Color::Black;
        return false;
    };

    for (const auto& [name, _] : matched_idx) {
        if (color[name] == Color::White) {
            if (dfs_visit(name)) {
                // Collect Gray nodes as cycle participants (approximate)
                std::string cycle_info;
                for (const auto& [n, c] : color) {
                    if (c == Color::Gray) {
                        if (!cycle_info.empty()) cycle_info += " → ";
                        cycle_info += n;
                    }
                }
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Rule_CyclicOverride,
                    "override cycle detected among: " + cycle_info,
                    std::source_location::current()});
            }
        }
    }

    // ---- 5. No cycles: eliminate overridden rules ----
    // A rule R is eliminated (matched = false) iff there exists another
    // matched rule M such that graph[M] contains R (M overrides R).
    for (const auto& [name, idx] : matched_idx) {
        bool is_overridden = false;
        for (const auto& [other_name, _] : matched_idx) {
            if (other_name == name) continue;
            const auto& edges = graph[other_name];
            if (std::find(edges.begin(), edges.end(), name) != edges.end()) {
                is_overridden = true;
                break;
            }
        }
        if (is_overridden) {
            evaluated[idx].matched = false;
        }
    }

    return {};
}

}  // namespace sai::rule::detail
