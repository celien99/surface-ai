#pragma once

#include <string>
#include <vector>

#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"

namespace sai::reasoner {

// -----------------------------------------------------------------------
// EvidenceItem — one piece of evidence in a reasoning decision
// -----------------------------------------------------------------------
struct EvidenceItem {
    std::string trace_id;          // links to a TraceStep ID
    std::string key;               // fact key or rule name
    rule::Value value;             // fact value or rule action severity
    rule::FactSource source;       // origin of this evidence
    std::string description;       // human-readable summary
};

// -----------------------------------------------------------------------
// EvidenceCollector — assemble evidence items from facts + trace + rules
// -----------------------------------------------------------------------
class EvidenceCollector {
public:
    /// Pack facts, trace steps, and resolved rules into an evidence array.
    /// Each fact entry is wrapped in an EvidenceItem and linked to the
    /// first trace step whose description mentions its key.
    static auto Pack(const rule::FactBase& facts,
                     const std::vector<rule::TraceStep>& trace,
                     const std::vector<rule::ResolvedRule>& rules)
        -> std::vector<EvidenceItem>;
};

}  // namespace sai::reasoner
