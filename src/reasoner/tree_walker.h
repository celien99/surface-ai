#pragma once

#include <string>

#include "sai/reasoner/decision_tree.h"
#include "sai/reasoner/trace_recorder.h"
#include "sai/rule/fact_base.h"

namespace sai::reasoner {

// -----------------------------------------------------------------------
// WalkResult — output of TreeWalker::Walk
// -----------------------------------------------------------------------
struct WalkResult {
    std::string label;
    double score{0.0};
    std::string recommendation;
};

// -----------------------------------------------------------------------
// TreeWalker — recursive decision-tree traversal
// -----------------------------------------------------------------------
class TreeWalker {
public:
    /// Walk the decision tree rooted at `node`, evaluating facts at each
    /// branch and scoring at each leaf. Records traversal steps into
    /// `tracer` as side effect.
    static auto Walk(const IDecisionNode& node,
                     const rule::FactBase& facts,
                     TraceRecorder& tracer) -> WalkResult;

private:
    /// Match a branch key against a numeric value. Supports ">X", ">=X",
    /// "<X", "<=X" operators. Returns false for non-numeric keys.
    static auto MatchNumeric(const std::string& key, double value) -> bool;
};

}  // namespace sai::reasoner
