#include "sai/reasoner/evidence_collector.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"

namespace sai::reasoner {

auto EvidenceCollector::Pack(
    const rule::FactBase& facts,
    const std::vector<rule::TraceStep>& trace,
    const std::vector<rule::ResolvedRule>& rules)
    -> std::vector<EvidenceItem> {
    std::vector<EvidenceItem> items;

    // Collect all fact entries and their sources.
    auto entries = facts.AllEntries();
    auto sources = facts.AllSources();

    // Build a map: fact key → first trace step ID whose description mentions
    // that key. This links each fact to its evaluating trace step.
    std::map<std::string, std::string> key_to_trace;
    for (auto& step : trace) {
        for (auto& [key, _] : entries) {
            if (step.description.find(key) != std::string::npos &&
                key_to_trace.find(key) == key_to_trace.end()) {
                key_to_trace[key] = step.id;
            }
        }
    }

    // Create one EvidenceItem per fact entry.
    for (auto& [key, val] : entries) {
        EvidenceItem item;
        item.key = key;
        item.value = val;
        item.description = "fact: " + key;

        // Find the corresponding source.
        auto src_it = std::find_if(sources.begin(), sources.end(),
                                   [&](auto& p) { return p.first == key; });
        if (src_it != sources.end()) {
            item.source = src_it->second;
        }

        // Link to trace step if one mentions this key.
        auto tr_it = key_to_trace.find(key);
        if (tr_it != key_to_trace.end()) {
            item.trace_id = tr_it->second;
        }

        items.push_back(std::move(item));
    }

    // Create one EvidenceItem per resolved rule with its action details.
    for (auto& rule : rules) {
        EvidenceItem item;
        item.key = rule.name;
        item.value = rule::Value::Of(rule.action.base_severity);
        item.description = "rule: " + rule.name +
                           (rule.matched ? " (matched)" : " (unmatched)");

        rule::FactSource src;
        src.kind = rule.matched ? rule::FactSourceKind::Computed
                                : rule::FactSourceKind::Default;
        src.description = "rule evaluation";
        item.source = src;

        // Link to the first eval trace step of this rule, if any.
        if (!rule.eval_trace.empty()) {
            item.trace_id = rule.eval_trace.front().id;
        }

        items.push_back(std::move(item));
    }

    return items;
}

}  // namespace sai::reasoner
