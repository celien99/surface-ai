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

// -----------------------------------------------------------------------
// RuleAction — the consequence matched by a rule
// -----------------------------------------------------------------------
struct RuleAction {
    std::string label;
    double base_severity{0.0};
    std::string recommendation;
};

// -----------------------------------------------------------------------
// TraceStep — one evaluation event in the diagnostic chain
// -----------------------------------------------------------------------
struct TraceStep {
    std::string id;
    enum class Level { Expression, Rule, TreeBranch, Scoring };
    Level level;
    std::string description;
    std::string source_location;
    std::chrono::microseconds timestamp;
    std::optional<std::string> parent_id;
};

// -----------------------------------------------------------------------
// Rule — one rule loaded from YAML, owning its parsed condition tree
// -----------------------------------------------------------------------
struct Rule {
    std::string name;
    uint32_t priority{0};
    std::string rule_set;
    std::vector<std::string> overrides;
    std::vector<std::string> overridden_by;
    std::unique_ptr<IExpression> condition;
    RuleAction action;
};

// -----------------------------------------------------------------------
// ResolvedRule — result of evaluating one rule (matched or not)
// -----------------------------------------------------------------------
struct ResolvedRule {
    std::string name;
    bool matched;
    RuleAction action;
    std::vector<TraceStep> eval_trace;
};

// -----------------------------------------------------------------------
// OverlapWarning — two rules within the same rule_set reference common facts
// -----------------------------------------------------------------------
struct OverlapWarning {
    std::string rule_a;
    std::string rule_b;
    std::vector<std::string> common_fields;
};

// -----------------------------------------------------------------------
// RuleEngine — loads YAML rule definitions, evaluates conditions against a
// FactBase, resolves override conflicts, and detects field-reference overlaps.
// -----------------------------------------------------------------------
class RuleEngine {
public:
    auto LoadFromYAML(std::filesystem::path) -> Result<void>;
    auto EnableHotReload(std::filesystem::path, std::stop_token) -> Result<void>;
    auto EvaluateAll(FactBase&) -> Result<std::vector<ResolvedRule>>;
    auto ResolveConflicts(const std::vector<ResolvedRule>&) -> std::vector<ResolvedRule>;
    auto DetectOverlaps() const -> std::vector<OverlapWarning>;

private:
    // Evaluate all rules in a single rule_set (called from EvaluateAll, possibly
    // in parallel across multiple rule_sets).
    auto EvaluateRuleSet(FactBase&, const std::vector<Rule>&) -> std::vector<ResolvedRule>;

    std::map<std::string, std::vector<Rule>> rule_sets_;
    std::filesystem::path current_path_;
};

}  // namespace sai::rule
