#include "sai/reasoner/trace_recorder.h"

#include <chrono>
#include <string>

#include "sai/rule/rule_engine.h"

namespace sai::reasoner {

auto TraceRecorder::NextId() -> std::string {
    return "step_" + std::to_string(next_id_++);
}

auto TraceRecorder::Record(rule::TraceStep::Level level, std::string desc,
                           std::string source) -> std::string {
    auto id = NextId();
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch());
    steps_.push_back({id, level, std::move(desc), std::move(source), ts,
                      std::nullopt});
    return id;
}

auto TraceRecorder::RecordExpression(std::string desc,
                                     std::string source) -> std::string {
    return Record(rule::TraceStep::Level::Expression, std::move(desc),
                  std::move(source));
}

auto TraceRecorder::RecordRule(std::string rule_name,
                               bool matched) -> std::string {
    auto desc = rule_name + " (matched=" + (matched ? "true" : "false") + ")";
    return Record(rule::TraceStep::Level::Rule, std::move(desc),
                  std::move(rule_name));
}

auto TraceRecorder::RecordTreeBranch(std::string field, std::string value,
                                     std::string branch) -> std::string {
    auto desc = field + "=" + value + " -> " + branch;
    return Record(rule::TraceStep::Level::TreeBranch, std::move(desc),
                  std::move(field));
}

auto TraceRecorder::RecordScoring(std::string formula_desc,
                                  double score) -> std::string {
    auto desc = formula_desc + " score=" + std::to_string(score);
    return Record(rule::TraceStep::Level::Scoring, std::move(desc),
                  std::move(formula_desc));
}

auto TraceRecorder::AllSteps() const -> const std::vector<rule::TraceStep>& {
    return steps_;
}

}  // namespace sai::reasoner
