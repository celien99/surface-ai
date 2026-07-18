#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sai/core/error.h"
#include "sai/core/service.h"
#include "sai/core/type_id.h"
#include "sai/reasoner/decision_tree.h"
#include "sai/reasoner/evidence_collector.h"
#include "sai/rule/fact_base.h"
#include "sai/rule/rule_engine.h"

namespace sai::reasoner {

// -----------------------------------------------------------------------
// ReasoningResult — output of the reasoning pipeline
// -----------------------------------------------------------------------
struct ReasoningResult {
    std::string surface_id;                  // M7: surface identifier (for multi-position pipelines)
    std::uint16_t position_id{0};           // M7: position index (for multi-position pipelines)
    std::string verdict;                     // NG / WARN / OK / UNCERTAIN
    double severity{0.0};                    // 0.0 – 1.0 (leaf score)
    std::string recommendation;              // human-readable action
    double confidence{0.0};                  // reuses leaf score
    std::vector<rule::TraceStep> trace;      // full decision trace
    std::vector<EvidenceItem> evidence;      // fact sources + rule actions
    std::vector<std::string> triggered_rules;
    std::vector<std::string> overridden_rules;
};

// -----------------------------------------------------------------------
// IReasoner — service interface for rule-based reasoning
// -----------------------------------------------------------------------
class IReasoner : public sai::IService {
public:
    SAI_DECLARE_TYPE_ID(sai::reasoner::IReasoner)
    /// Evaluate facts through the decision tree, merge rule results, and
    /// produce a complete ReasoningResult (verdict, severity, trace,
    /// evidence, triggered/overridden rule lists).
    virtual auto Reason(
        const rule::FactBase& facts,
        const std::vector<rule::ResolvedRule>& rules
    ) -> Result<ReasoningResult> = 0;

    /// M7: Hot-reload the decision tree from a YAML file.
    /// Called by ConfigViewModel::ApplyTreeChanges. On success, the next
    /// Reason() call uses the new tree. On failure, the old tree is retained.
    virtual auto ReloadTree(std::filesystem::path path) -> Result<void> = 0;
};

// -----------------------------------------------------------------------
// DefaultReasoner — concrete IReasoner with a single DecisionTree
// -----------------------------------------------------------------------
class DefaultReasoner final : public IReasoner {
public:
    explicit DefaultReasoner(std::unique_ptr<DecisionTree> tree);
    ~DefaultReasoner() override = default;

    SAI_DECLARE_TYPE_ID(sai::reasoner::DefaultReasoner)

    auto Reason(const rule::FactBase& facts,
                const std::vector<rule::ResolvedRule>& rules)
        -> Result<ReasoningResult> override;

    auto ReloadTree(std::filesystem::path path) -> Result<void> override;

private:
    std::unique_ptr<DecisionTree> tree_;
};

}  // namespace sai::reasoner
