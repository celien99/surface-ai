#include "sai/reasoner/decision_tree.h"

#include <source_location>
#include <system_error>
#include <utility>

#include <yaml-cpp/yaml.h>

#include "sai/core/error.h"

namespace sai::reasoner {

// ===========================================================================
// BranchNode
// ===========================================================================

auto BranchNode::FieldName() const -> std::string_view { return field_; }

auto BranchNode::Branches()
    const -> const std::map<std::string, std::unique_ptr<IDecisionNode>>& {
    return branches_;
}

auto BranchNode::DefaultBranch() const -> const IDecisionNode* {
    return default_.get();
}

auto BranchNode::SetField(std::string field) -> void {
    field_ = std::move(field);
}

auto BranchNode::AddBranch(std::string match_value,
                           std::unique_ptr<IDecisionNode> node) -> void {
    branches_.emplace(std::move(match_value), std::move(node));
}

auto BranchNode::SetDefault(std::unique_ptr<IDecisionNode> node) -> void {
    default_ = std::move(node);
}

// ===========================================================================
// LeafNode
// ===========================================================================

auto LeafNode::Formulas() const -> const std::vector<ScoreFormula>& {
    return formulas_;
}

auto LeafNode::Label() const -> std::string_view { return label_; }

auto LeafNode::Recommendation() const -> std::string_view {
    return recommendation_;
}

auto LeafNode::SetLabel(std::string label) -> void {
    label_ = std::move(label);
}

auto LeafNode::SetRecommendation(std::string rec) -> void {
    recommendation_ = std::move(rec);
}

auto LeafNode::AddFormula(ScoreFormula formula) -> void {
    formulas_.push_back(std::move(formula));
}

// ===========================================================================
// DecisionTree — private ParseNode (recursive YAML → node construction)
// ===========================================================================

auto DecisionTree::ParseNode(const YAML::Node& yaml)
    -> Result<std::unique_ptr<IDecisionNode>> {
    auto type_node = yaml["type"];
    if (!type_node) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Reasoner_InvalidTree,
            "node missing required 'type' field",
            std::source_location::current()});
    }

    auto type = type_node.as<std::string>();

    if (type == "branch") {
        auto branch = std::make_unique<BranchNode>();

        // Parse field name — every branch must name the fact field it inspects.
        auto field_node = yaml["field"];
        if (!field_node) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Reasoner_InvalidTree,
                "branch node missing required 'field'",
                std::source_location::current()});
        }
        branch->SetField(field_node.as<std::string>());

        // Parse named branches map.
        if (auto branches_node = yaml["branches"]) {
            if (branches_node.IsMap()) {
                for (auto it = branches_node.begin(); it != branches_node.end();
                     ++it) {
                    auto key = it->first.as<std::string>();
                    auto child = ParseNode(it->second);
                    if (!child) {
                        return tl::make_unexpected(child.error());
                    }
                    branch->AddBranch(std::move(key), std::move(*child));
                }
            }
        }

        // Parse default child — every branch MUST have a default.
        if (auto default_node = yaml["default"]) {
            auto child = ParseNode(default_node);
            if (!child) {
                return tl::make_unexpected(child.error());
            }
            branch->SetDefault(std::move(*child));
        } else {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Reasoner_InvalidTree,
                "branch node missing required 'default' child",
                std::source_location::current()});
        }

        return branch;
    }

    if (type == "leaf") {
        auto leaf = std::make_unique<LeafNode>();

        if (auto label_node = yaml["label"]) {
            leaf->SetLabel(label_node.as<std::string>());
        }
        if (auto rec_node = yaml["recommendation"]) {
            leaf->SetRecommendation(rec_node.as<std::string>());
        }

        // Parse formulas array.
        if (auto formulas_node = yaml["formulas"]) {
            if (formulas_node.IsSequence()) {
                for (std::size_t i = 0; i < formulas_node.size(); ++i) {
                    auto f_node = formulas_node[i];
                    ScoreFormula formula;

                    if (auto w = f_node["weights"]) {
                        for (auto it = w.begin(); it != w.end(); ++it) {
                            formula.weights.push_back(it->as<double>());
                        }
                    }
                    if (auto feat = f_node["features"]) {
                        for (auto it = feat.begin(); it != feat.end(); ++it) {
                            formula.features.push_back(it->as<std::string>());
                        }
                    }
                    formula.threshold =
                        f_node["threshold"].as<double>(0.0);

                    leaf->AddFormula(std::move(formula));
                }
            }
        }

        return leaf;
    }

    return tl::make_unexpected(ErrorInfo{
        ErrorCode::Reasoner_InvalidTree,
        "unknown node type '" + type + "' (expected 'branch' or 'leaf')",
        std::source_location::current()});
}

// ===========================================================================
// DecisionTree — public API
// ===========================================================================

DecisionTree::DecisionTree(std::unique_ptr<IDecisionNode> root)
    : root_(std::move(root)) {}

auto DecisionTree::Root() const -> const IDecisionNode& { return *root_; }

auto DecisionTree::LoadFromYAML(std::filesystem::path path)
    -> Result<std::unique_ptr<DecisionTree>> {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Reasoner_TreeLoadFailed,
            "failed to load decision tree from " + path.string() +
                ": " + e.what(),
            std::source_location::current()});
    }

    auto node = ParseNode(root);
    if (!node) {
        return tl::make_unexpected(node.error());
    }

    return std::unique_ptr<DecisionTree>(
        new DecisionTree(std::move(*node)));
}

}  // namespace sai::reasoner
