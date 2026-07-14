#include "tree_walker.h"

#include <string>

#include "sai/reasoner/score_calculator.h"
#include "sai/reasoner/trace_recorder.h"

namespace sai::reasoner {

auto TreeWalker::Walk(const IDecisionNode& node,
                       const rule::FactBase& facts,
                       TraceRecorder& tracer) -> WalkResult {
    // --- Leaf node: evaluate formulas and return score ---
    if (node.GetKind() == IDecisionNode::Kind::Leaf) {
        auto& leaf = static_cast<const LeafNode&>(node);
        double score = ScoreCalculator::ComputeMax(leaf.Formulas(), facts);
        tracer.RecordScoring(
            std::string("Leaf: ") + std::string(leaf.Label()) +
                " score=" + std::to_string(score),
            score);
        return {std::string(leaf.Label()), score,
                std::string(leaf.Recommendation())};
    }

    // --- Branch node: dispatch by fact field value ---
    auto& branch = static_cast<const BranchNode&>(node);
    auto field_val = facts.Get(branch.FieldName());

    if (field_val.has_value()) {
        // Try string match first.
        if (auto s = field_val->AsString()) {
            auto it = branch.Branches().find(std::string(*s));
            if (it != branch.Branches().end()) {
                tracer.RecordTreeBranch(std::string(branch.FieldName()),
                                         std::string(*s), std::string(*s));
                return Walk(*it->second, facts, tracer);
            }
        }

        // Try numeric match (">10", "<5", ">=3", "<=8").
        if (auto d = field_val->AsDouble()) {
            for (auto& [key, child] : branch.Branches()) {
                if (MatchNumeric(key, *d)) {
                    tracer.RecordTreeBranch(std::string(branch.FieldName()),
                                             std::to_string(*d), key);
                    return Walk(*child, facts, tracer);
                }
            }
        }
    }

    // --- Fallback to default branch ---
    tracer.RecordTreeBranch(
        std::string(branch.FieldName()),
        field_val.has_value() ? "value" : "missing",
        "default");

    if (!branch.DefaultBranch()) {
        // Should never happen — DecisionTree::ParseNode enforces that every
        // branch node has a default child. Defensive fallback only.
        return {"UNCERTAIN", 0.0, "No matching branch and no default"};
    }
    return Walk(*branch.DefaultBranch(), facts, tracer);
}

bool TreeWalker::MatchNumeric(const std::string& key, double value) {
    if (key.empty()) return false;

    char op = key[0];
    double threshold = 0.0;
    try {
        // Determine offset: 1 for single-char op, 2 for two-char op (>=, <=).
        size_t offset = (key.size() > 1 && key[1] == '=') ? 2 : 1;
        if (offset >= key.size()) return false;
        threshold = std::stod(key.substr(offset));
    } catch (...) {
        return false;
    }

    switch (op) {
    case '>':
        return (key.size() > 1 && key[1] == '=') ? value >= threshold
                                                   : value > threshold;
    case '<':
        return (key.size() > 1 && key[1] == '=') ? value <= threshold
                                                   : value < threshold;
    default:
        return false;
    }
}

}  // namespace sai::reasoner
