#include "sai/rule/rule_engine.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <set>
#include <sstream>
#include <string>
#include <system_error>

#include <yaml-cpp/yaml.h>

#include "sai/infra/logger.h"
#include "conflict_resolver.h"
#include "lexer.h"
#include "parser.h"

namespace sai::rule {

// ===========================================================================
// LoadFromYAML
// ===========================================================================

auto RuleEngine::LoadFromYAML(std::filesystem::path path) -> Result<void> {
    try {
        auto root = YAML::LoadFile(path.string());

        auto rs_node = root["rule_sets"];
        if (!rs_node) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Rule_ParseError,
                "missing top-level 'rule_sets' key in " + path.string(),
                std::source_location::current()});
        }

        std::map<std::string, std::vector<Rule>> loaded;

        for (auto rs_it = rs_node.begin(); rs_it != rs_node.end(); ++rs_it) {
            auto set_name = rs_it->first.as<std::string>();
            auto rules_seq = rs_it->second;
            if (!rules_seq.IsSequence()) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Rule_ParseError,
                    "rule_set '" + set_name + "' is not a sequence in " + path.string(),
                    std::source_location::current()});
            }

            std::vector<Rule> rules;
            rules.reserve(rules_seq.size());

            for (std::size_t ri = 0; ri < rules_seq.size(); ++ri) {
                auto rn = rules_seq[ri];
                Rule rule;

                rule.name = rn["name"].as<std::string>("");
                if (rule.name.empty()) {
                    return tl::make_unexpected(ErrorInfo{
                        ErrorCode::Rule_ParseError,
                        "rule at index " + std::to_string(ri) +
                            " in rule_set '" + set_name + "' has empty or missing 'name'",
                        std::source_location::current()});
                }

                rule.priority = rn["priority"].as<uint32_t>(0);
                rule.rule_set = set_name;

                // --- Parse condition string via Lexer + Parser ---
                auto cond_str = rn["condition"].as<std::string>("");
                if (!cond_str.empty()) {
                    Lexer lexer(cond_str);
                    auto tokens = lexer.Tokenize();
                    if (!tokens) {
                        return tl::make_unexpected(ErrorInfo{
                            ErrorCode::Rule_ParseError,
                            "lexer error for rule '" + rule.name + "': " + tokens.error().message,
                            std::source_location::current()});
                    }

                    Parser parser(std::move(*tokens));
                    auto expr = parser.Parse();
                    if (!expr) {
                        return tl::make_unexpected(ErrorInfo{
                            ErrorCode::Rule_ParseError,
                            "parser error for rule '" + rule.name + "': " + expr.error().message,
                            std::source_location::current()});
                    }

                    rule.condition = std::move(*expr);
                }

                // --- Action ---
                auto act = rn["action"];
                if (act) {
                    rule.action.label = act["label"].as<std::string>("");
                    rule.action.base_severity = act["base_severity"].as<double>(0.0);
                    rule.action.recommendation = act["recommendation"].as<std::string>("");
                }

                // --- Overrides / overridden_by ---
                if (auto ov = rn["overrides"]) {
                    for (auto o : ov) {
                        rule.overrides.push_back(o.as<std::string>());
                    }
                }
                if (auto ob = rn["overridden_by"]) {
                    for (auto o : ob) {
                        rule.overridden_by.push_back(o.as<std::string>());
                    }
                }

                rules.push_back(std::move(rule));
            }

            loaded[set_name] = std::move(rules);
        }

        rule_sets_ = std::move(loaded);
        current_path_ = std::move(path);
        return {};

    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Rule_ParseError,
            "YAML exception: " + std::string(e.what()),
            std::source_location::current()});
    }
}

// ===========================================================================
// EvaluateRuleSet (single rule_set, not thread-safe on FactBase)
// ===========================================================================

auto RuleEngine::EvaluateRuleSet(FactBase& facts, const std::vector<Rule>& rules)
    -> std::vector<ResolvedRule> {
    std::vector<ResolvedRule> results;
    results.reserve(rules.size());

    for (const auto& rule : rules) {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::microseconds>(now);

        TraceStep step;
        step.id = rule.name + ":eval";
        step.level = TraceStep::Level::Rule;
        step.description = "Evaluating condition for " + rule.name;
        step.source_location = rule.name;
        step.timestamp = timestamp;

        bool matched = false;

        if (rule.condition) {
            auto result = rule.condition->Evaluate(facts);
            if (result.has_value()) {
                matched = result->AsBool().value_or(false);
                step.description +=
                    matched ? " -> MATCH" : " -> NO_MATCH";
            } else {
                // Error recovery: skip this rule gracefully, do not crash.
                step.description +=
                    " -> ERROR: " + result.error().message;
                matched = false;
            }
        } else {
            // No condition → always matched.
            step.description += " -> MATCH (no condition)";
            matched = true;
        }

        std::vector<TraceStep> trace;
        trace.push_back(std::move(step));

        results.push_back(ResolvedRule{
            rule.name, matched, rule.action, std::move(trace)});
    }

    return results;
}

// ===========================================================================
// EvaluateAll  — parallel evaluation across rule_sets
// ===========================================================================

auto RuleEngine::EvaluateAll(FactBase& facts)
    -> Result<std::vector<ResolvedRule>> {
    if (rule_sets_.empty()) {
        return std::vector<ResolvedRule>{};
    }

    // Each rule_set is dispatched to std::async with its own FactBase
    // snapshot to avoid concurrent writes on the same FactBase (memoization).
    // The snapshot copies are cheap for small-to-medium FactBases.
    std::vector<std::future<std::vector<ResolvedRule>>> futures;
    futures.reserve(rule_sets_.size());

    for (auto it = rule_sets_.begin(); it != rule_sets_.end(); ++it) {
        futures.push_back(
            std::async(std::launch::async,
                       [this, fb = facts.Snapshot(), rules_ptr = &it->second]() mutable {
                           return EvaluateRuleSet(fb, *rules_ptr);
                       }));
    }

    std::vector<ResolvedRule> all;
    for (auto& f : futures) {
        auto batch = f.get();
        all.insert(all.end(), std::make_move_iterator(batch.begin()),
                   std::make_move_iterator(batch.end()));
    }

    return all;
}

// ===========================================================================
// ResolveConflicts  —  DAG-based override elimination
// ===========================================================================

auto RuleEngine::ResolveConflicts(
    const std::vector<ResolvedRule>& evaluated)
    -> std::vector<ResolvedRule> {
    // Work on a mutable copy so the internal function can modify matched flags.
    auto result = evaluated;

    auto status =
        detail::ResolveOverrideConflicts(rule_sets_, result);
    if (!status) {
        // Cycle detected — log a warning, return the input unchanged.
        sai::infra::Logger::Get("rule")
            .Log(sai::infra::LogLevel::Warning,
                 "ResolveConflicts: override cycle detected — {}",
                 status.error().message);
        return evaluated;
    }

    return result;
}

// ===========================================================================
// DetectOverlaps  —  O(n²) field-ref intersection within each rule_set
// ===========================================================================

auto RuleEngine::DetectOverlaps() const -> std::vector<OverlapWarning> {
    std::vector<OverlapWarning> warnings;

    for (const auto& [set_name, rules] : rule_sets_) {
        (void)set_name;

        // Collect field refs per rule (cache)
        std::vector<std::vector<std::string>> refs_by_rule;
        refs_by_rule.reserve(rules.size());
        for (const auto& rule : rules) {
            if (rule.condition) {
                refs_by_rule.push_back(rule.condition->CollectFieldRefs());
            } else {
                refs_by_rule.emplace_back();
            }
        }

        // O(n²) pairwise comparison within the same rule_set
        for (std::size_t i = 0; i < rules.size(); ++i) {
            for (std::size_t j = i + 1; j < rules.size(); ++j) {
                // Sort both lists and set_intersect
                std::vector<std::string> a = refs_by_rule[i];
                std::vector<std::string> b = refs_by_rule[j];
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());

                std::vector<std::string> common;
                std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                                      std::back_inserter(common));

                if (!common.empty()) {
                    OverlapWarning ow;
                    ow.rule_a = rules[i].name;
                    ow.rule_b = rules[j].name;
                    ow.common_fields = std::move(common);
                    warnings.push_back(std::move(ow));
                }
            }
        }
    }

    return warnings;
}

// ===========================================================================
// EnableHotReload  —  Linux: inotify; macOS: no-op
// ===========================================================================

auto RuleEngine::EnableHotReload(std::filesystem::path /*path*/,
                                  std::stop_token /*token*/) -> Result<void> {
    // TODO(yyh): Linux-gated hot-reload via inotify.
    //   On Linux: watch current_path_ for IN_CLOSE_WRITE using inotify;
    //   on each event call LoadFromYAML(current_path_); on parse failure log
    //   and keep the previous rule set. Use stop_token to shut down the
    //   watch thread.
    // On macOS this is intentionally a no-op — inotify is Linux-specific.
    return {};
}

}  // namespace sai::rule
