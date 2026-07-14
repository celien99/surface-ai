#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sai/rule/rule_engine.h"
#include "src/rule/conflict_resolver.h"

namespace sai::rule {
namespace {

// ===========================================================================
// Helper — build a vector of ResolvedRule from name+matched pairs
// ===========================================================================
auto MakeResolved(std::initializer_list<std::pair<std::string, bool>> pairs)
    -> std::vector<ResolvedRule> {
    std::vector<ResolvedRule> result;
    for (const auto& [name, matched] : pairs) {
        result.push_back(
            {name, matched, {"", 0.0, ""}, {}});
    }
    return result;
}

// Helper — add rules to a rule_set map. Rule has unique_ptr so it's
// move-only and cannot use initializer_list assignment.
auto AddRules(std::map<std::string, std::vector<Rule>>& sets,
              const std::string& set_name,
              std::vector<Rule> rules) -> void {
    sets[set_name] = std::move(rules);
}

// ===========================================================================
// HigherPriorityWinsWithoutOverrides
// Both matched, no overrides → both survive (priority alone is not a conflict
// reason per spec; overrides are explicit).
// ===========================================================================

TEST(ConflictResolverTest, HigherPriorityWinsWithoutOverrides) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b;
        a.name = "A";
        a.priority = 200;
        b.name = "B";
        b.priority = 50;
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved({{"A", true}, {"B", true}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Both survive because there are no explicit override declarations
    int matched = 0;
    for (const auto& r : evaluated) {
        if (r.matched) ++matched;
    }
    EXPECT_EQ(matched, 2);
}

// ===========================================================================
// ExplicitOverrideEliminatesTarget
// A.overrides = ["B"], both matched → B is eliminated
// ===========================================================================

TEST(ConflictResolverTest, ExplicitOverrideEliminatesTarget) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b;
        a.name = "A";
        a.overrides = {"B"};
        b.name = "B";
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved({{"A", true}, {"B", true}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // A survives (matched), B eliminated (matched → false)
    for (const auto& r : evaluated) {
        if (r.name == "A") EXPECT_TRUE(r.matched);
        if (r.name == "B") EXPECT_FALSE(r.matched);
    }
}

// ===========================================================================
// OverriddenBySymmetrical
// A.overrides=["B"], B.overridden_by=["A"] → B eliminated (same effect)
// ===========================================================================

TEST(ConflictResolverTest, OverriddenBySymmetrical) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b;
        a.name = "A";
        a.overrides = {"B"};
        b.name = "B";
        b.overridden_by = {"A"};
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved({{"A", true}, {"B", true}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    for (const auto& r : evaluated) {
        if (r.name == "A") EXPECT_TRUE(r.matched);
        if (r.name == "B") EXPECT_FALSE(r.matched);
    }
}

// ===========================================================================
// CyclicOverrideDetected
// A.overrides=["B"], B.overrides=["C"], C.overrides=["A"] → error
// ===========================================================================

TEST(ConflictResolverTest, CyclicOverrideDetected) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b, c;
        a.name = "A";
        a.overrides = {"B"};
        b.name = "B";
        b.overrides = {"C"};
        c.name = "C";
        c.overrides = {"A"};
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rules.push_back(std::move(c));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved(
        {{"A", true}, {"B", true}, {"C", true}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Rule_CyclicOverride);
}

// ===========================================================================
// NonMatchedRuleNotOverridden
// A.overrides=["B"], A matched, B NOT matched → B stays unmatched (no crash)
// ===========================================================================

TEST(ConflictResolverTest, NonMatchedRuleNotOverridden) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b;
        a.name = "A";
        a.overrides = {"B"};
        b.name = "B";
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved({{"A", true}, {"B", false}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // A matched, B stays unmatched (unchanged)
    for (const auto& r : evaluated) {
        if (r.name == "A") EXPECT_TRUE(r.matched);
        if (r.name == "B") EXPECT_FALSE(r.matched);
    }
}

// ===========================================================================
// NoMatchedRules → no-op
// ===========================================================================

TEST(ConflictResolverTest, NoMatchedRules) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> rules;
        Rule a, b;
        a.name = "A";
        a.overrides = {"B"};
        b.name = "B";
        rules.push_back(std::move(a));
        rules.push_back(std::move(b));
        rule_sets["test"] = std::move(rules);
    }

    auto evaluated = MakeResolved({{"A", false}, {"B", false}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value());

    // Both still unmatched
    for (const auto& r : evaluated) {
        EXPECT_FALSE(r.matched);
    }
}

// ===========================================================================
// MultiSetOverride — rule from one set overrides rule from another set
// ===========================================================================

TEST(ConflictResolverTest, MultiSetOverride) {
    std::map<std::string, std::vector<Rule>> rule_sets;

    {
        std::vector<Rule> primary, secondary;
        Rule a, b;
        a.name = "A";
        a.overrides = {"B"};
        primary.push_back(std::move(a));
        b.name = "B";
        secondary.push_back(std::move(b));
        rule_sets["primary"] = std::move(primary);
        rule_sets["secondary"] = std::move(secondary);
    }

    auto evaluated = MakeResolved({{"A", true}, {"B", true}});

    auto result = detail::ResolveOverrideConflicts(rule_sets, evaluated);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    for (const auto& r : evaluated) {
        if (r.name == "A") EXPECT_TRUE(r.matched);
        if (r.name == "B") EXPECT_FALSE(r.matched);
    }
}

}  // namespace
}  // namespace sai::rule
