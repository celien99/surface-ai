#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sai/rule/rule_engine.h"  // TraceStep

namespace sai::reasoner {

// -----------------------------------------------------------------------
// TraceRecorder — accumulate TraceStep objects during reasoning traversal
// -----------------------------------------------------------------------
class TraceRecorder {
public:
    TraceRecorder() = default;

    /// Record an expression evaluation step. Returns the step ID.
    auto RecordExpression(std::string desc, std::string source) -> std::string;

    /// Record a rule match/mismatch step. Returns the step ID.
    auto RecordRule(std::string rule_name, bool matched) -> std::string;

    /// Record a decision-tree branch dispatch step. Returns the step ID.
    auto RecordTreeBranch(std::string field, std::string value,
                          std::string branch) -> std::string;

    /// Record a scoring evaluation step. Returns the step ID.
    auto RecordScoring(std::string formula_desc, double score) -> std::string;

    /// Return all accumulated trace steps (in insertion order).
    auto AllSteps() const -> const std::vector<rule::TraceStep>&;

private:
    auto NextId() -> std::string;
    auto Record(rule::TraceStep::Level level, std::string desc,
                std::string source) -> std::string;

    std::vector<rule::TraceStep> steps_;
    uint64_t next_id_{0};
};

}  // namespace sai::reasoner
