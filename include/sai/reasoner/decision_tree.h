#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "sai/core/object.h"
#include "sai/core/error.h"

namespace sai::reasoner {

// -----------------------------------------------------------------------
// ScoreFormula — one weighted-sum formula evaluated against fact fields
// -----------------------------------------------------------------------
struct ScoreFormula {
    std::vector<double> weights;
    std::vector<std::string> features;
    double threshold{0.0};
};

// -----------------------------------------------------------------------
// VerdictMapping — score → verdict boundary (hot-reloadable via YAML)
// -----------------------------------------------------------------------
struct VerdictMapping {
    double ng_threshold{0.7};    // score > this → "NG"
    double warn_threshold{0.3};  // ng_threshold ≥ score > this → "WARN"
    // score ≤ warn_threshold → "OK"
};

// -----------------------------------------------------------------------
// IDecisionNode — polymorphic tree node (Branch or Leaf)
// -----------------------------------------------------------------------
class IDecisionNode : public sai::Object {
public:
    enum class Kind { Branch, Leaf };
    virtual auto GetKind() const noexcept -> Kind = 0;
};

// -----------------------------------------------------------------------
// BranchNode — dispatches to a child by evaluating a field value
// -----------------------------------------------------------------------
class BranchNode final : public IDecisionNode {
public:
    auto GetKind() const noexcept -> Kind override { return Kind::Branch; }
    auto FieldName() const -> std::string_view;
    auto Branches() const -> const std::map<std::string, std::unique_ptr<IDecisionNode>>&;
    auto DefaultBranch() const -> const IDecisionNode*;

    // Builder API (used by DecisionTree::LoadFromYAML)
    auto SetField(std::string field) -> void;
    auto AddBranch(std::string match_value, std::unique_ptr<IDecisionNode>) -> void;
    auto SetDefault(std::unique_ptr<IDecisionNode>) -> void;

private:
    std::string field_;
    std::map<std::string, std::unique_ptr<IDecisionNode>> branches_;
    std::unique_ptr<IDecisionNode> default_;
};

// -----------------------------------------------------------------------
// LeafNode — terminal node with score formulas and a decision label
// -----------------------------------------------------------------------
class LeafNode final : public IDecisionNode {
public:
    auto GetKind() const noexcept -> Kind override { return Kind::Leaf; }
    auto Formulas() const -> const std::vector<ScoreFormula>&;
    auto Label() const -> std::string_view;
    auto Recommendation() const -> std::string_view;

    auto SetLabel(std::string) -> void;
    auto SetRecommendation(std::string) -> void;
    auto AddFormula(ScoreFormula) -> void;

private:
    std::vector<ScoreFormula> formulas_;
    std::string label_;
    std::string recommendation_;
};

// -----------------------------------------------------------------------
// DecisionTree — root-owning entry point, constructed from YAML
// -----------------------------------------------------------------------
class DecisionTree {
public:
    static auto LoadFromYAML(std::filesystem::path) -> Result<std::unique_ptr<DecisionTree>>;
    auto Root() const -> const IDecisionNode&;
    auto VerdictMapping() const -> const reasoner::VerdictMapping&;

    explicit DecisionTree(std::unique_ptr<IDecisionNode> root);

private:
    static auto ParseNode(const YAML::Node& yaml) -> Result<std::unique_ptr<IDecisionNode>>;
    std::unique_ptr<IDecisionNode> root_;
    struct VerdictMapping verdict_mapping_;
};

}  // namespace sai::reasoner
