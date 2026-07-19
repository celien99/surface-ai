// rule_eval_output.h — RuleEvalOutput struct extracted from stage_node.h
// to break the std::variant compile-time coupling (Batch T3 refactor).
//
// Only included by stages that produce or consume RuleEvalOutput:
// rule_eval_stage.cpp (producer), reason_stage.cpp (consumer).
#pragma once

#include <vector>

#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>

namespace sai::pipeline {

struct RuleEvalOutput {
    sai::rule::FactBase facts;
    std::vector<sai::rule::ResolvedRule> rules;
};

}  // namespace sai::pipeline
