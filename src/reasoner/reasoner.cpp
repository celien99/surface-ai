#include "sai/reasoner/reasoner.h"

#include <string>
#include <vector>

#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"
#include "sai/reasoner/decision_tree.h"
#include "tree_walker.h"
#include "trace_recorder.h"
#include "sai/reasoner/evidence_collector.h"

namespace sai::reasoner {

// ===========================================================================
// IReasoner / DefaultReasoner
// ===========================================================================

DefaultReasoner::DefaultReasoner(std::unique_ptr<DecisionTree> tree)
    : tree_(std::move(tree)) {}

auto DefaultReasoner::Reason(
    const rule::FactBase& facts,
    const std::vector<rule::ResolvedRule>& rules
) -> Result<ReasoningResult> {
    TraceRecorder tracer;
    EvidenceCollector evidence;

    // --- 1. Merge rule results into the trace ---
    std::vector<std::string> triggered, overridden;
    for (auto& r : rules) {
        if (r.matched) {
            triggered.push_back(r.name);
            tracer.RecordRule(r.name, true);
            // Inject individual eval_trace steps under the rule trace entry.
            for (auto& step : r.eval_trace) {
                tracer.RecordExpression(step.description, step.source_location);
            }
        } else {
            overridden.push_back(r.name);
        }
    }

    // --- 2. Walk the decision tree ---
    auto walk_result = TreeWalker::Walk(tree_->Root(), facts, tracer);

    // --- 3. Determine verdict from label or score ---
    std::string verdict;
    if (walk_result.label == "NG" || walk_result.label == "WARN" ||
        walk_result.label == "OK" || walk_result.label == "UNCERTAIN") {
        verdict = walk_result.label;
    } else if (walk_result.score > 0.7) {
        verdict = "NG";
    } else if (walk_result.score > 0.3) {
        verdict = "WARN";
    } else {
        verdict = "OK";
    }

    // --- 4. Assemble full result ---
    ReasoningResult result;
    result.verdict = verdict;
    result.severity = walk_result.score;
    result.recommendation = walk_result.recommendation;
    result.confidence = walk_result.score;  // reuse leaf score as confidence
    result.trace = tracer.AllSteps();
    result.evidence = evidence.Pack(facts, tracer.AllSteps(), rules);
    result.triggered_rules = std::move(triggered);
    result.overridden_rules = std::move(overridden);

    return result;
}

}  // namespace sai::reasoner
