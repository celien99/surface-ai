#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "sai/reasoner/decision_tree.h"

namespace sai::reasoner {
namespace {

// -----------------------------------------------------------------------
// Helper — write a YAML string to a temporary file and return its path.
// -----------------------------------------------------------------------
auto TempYAMLPath(const std::string& content) -> std::filesystem::path {
    static std::size_t counter = 0;
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / ("decision_tree_test_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

// ===========================================================================
// LoadFromYAML_BasicTree
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_BasicTree) {
    auto yaml = R"(
type: branch
field: severity
branches:
  low:
    type: leaf
    label: low_severity
    recommendation: monitor
  high:
    type: leaf
    label: high_severity
    recommendation: inspect
default:
  type: leaf
  label: unknown_severity
  recommendation: check
)";

    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto& tree = *result.value();
    auto& root = tree.Root();
    EXPECT_EQ(root.GetKind(), IDecisionNode::Kind::Branch);

    auto& branch = static_cast<const BranchNode&>(root);
    EXPECT_EQ(branch.FieldName(), "severity");
    EXPECT_EQ(branch.Branches().size(), 2);

    auto* def = branch.DefaultBranch();
    ASSERT_NE(def, nullptr);
    EXPECT_EQ(def->GetKind(), IDecisionNode::Kind::Leaf);
    EXPECT_EQ(static_cast<const LeafNode&>(*def).Label(), "unknown_severity");

    std::filesystem::remove(path);
}

// ===========================================================================
// LoadFromYAML_MultiLevel  —  branch → branch → leaf nesting
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_MultiLevel) {
    auto yaml = R"(
type: branch
field: scratch_area
branches:
  ">10":
    type: leaf
    label: reject
    recommendation: scrap_part
  "<5":
    type: leaf
    label: accept
    recommendation: continue
default:
  type: branch
  field: scratch_depth
  branches:
    ">0.5":
      type: leaf
      label: conditional_pass
      recommendation: rework
  default:
    type: leaf
    label: accept
    recommendation: continue
)";

    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto& root = static_cast<const BranchNode&>(result.value()->Root());
    EXPECT_EQ(root.FieldName(), "scratch_area");
    EXPECT_EQ(root.Branches().size(), 2);

    // Verify numeric branch keys stored as plain strings.
    auto& br = root.Branches();
    EXPECT_NE(br.find(">10"), br.end());
    EXPECT_NE(br.find("<5"), br.end());

    // Default is a nested BranchNode.
    auto* def = root.DefaultBranch();
    ASSERT_NE(def, nullptr);
    ASSERT_EQ(def->GetKind(), IDecisionNode::Kind::Branch);

    auto& nested = static_cast<const BranchNode&>(*def);
    EXPECT_EQ(nested.FieldName(), "scratch_depth");
    EXPECT_EQ(nested.Branches().size(), 1);
    EXPECT_NE(nested.Branches().find(">0.5"), nested.Branches().end());
    ASSERT_NE(nested.DefaultBranch(), nullptr);
    EXPECT_EQ(static_cast<const LeafNode&>(*nested.DefaultBranch()).Label(),
              "accept");

    std::filesystem::remove(path);
}

// ===========================================================================
// LoadFromYAML_MissingDefault  —  branch without default → InvalidTree
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_MissingDefault) {
    auto yaml = R"(
type: branch
field: severity
branches:
  low:
    type: leaf
    label: low_severity
)";

    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Reasoner_InvalidTree);

    std::filesystem::remove(path);
}

// ===========================================================================
// LoadFromYAML_LeafWithFormulas  —  ScoreFormula parsing
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_LeafWithFormulas) {
    auto yaml = R"(
type: leaf
label: scratch_detected
recommendation: inspect_region
formulas:
  - weights: [0.5, 0.3, 0.2]
    features: [depth, width, contrast]
    threshold: 0.7
  - weights: [0.8, 0.2]
    features: [area, elongation]
    threshold: 0.5
)";

    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto& leaf =
        static_cast<const LeafNode&>(result.value()->Root());
    EXPECT_EQ(leaf.Label(), "scratch_detected");
    EXPECT_EQ(leaf.Recommendation(), "inspect_region");

    auto& formulas = leaf.Formulas();
    ASSERT_EQ(formulas.size(), 2);

    // --- First formula ---
    EXPECT_EQ(formulas[0].weights.size(), 3);
    EXPECT_DOUBLE_EQ(formulas[0].weights[0], 0.5);
    EXPECT_DOUBLE_EQ(formulas[0].weights[1], 0.3);
    EXPECT_DOUBLE_EQ(formulas[0].weights[2], 0.2);

    EXPECT_EQ(formulas[0].features.size(), 3);
    EXPECT_EQ(formulas[0].features[0], "depth");
    EXPECT_EQ(formulas[0].features[1], "width");
    EXPECT_EQ(formulas[0].features[2], "contrast");

    EXPECT_DOUBLE_EQ(formulas[0].threshold, 0.7);

    // --- Second formula ---
    EXPECT_EQ(formulas[1].weights.size(), 2);
    EXPECT_DOUBLE_EQ(formulas[1].weights[0], 0.8);
    EXPECT_DOUBLE_EQ(formulas[1].weights[1], 0.2);

    EXPECT_EQ(formulas[1].features.size(), 2);
    EXPECT_EQ(formulas[1].features[0], "area");
    EXPECT_EQ(formulas[1].features[1], "elongation");

    EXPECT_DOUBLE_EQ(formulas[1].threshold, 0.5);

    std::filesystem::remove(path);
}

// ===========================================================================
// LoadFromYAML_FileNotFound  —  non-existent path → TreeLoadFailed
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_FileNotFound) {
    auto path = std::filesystem::temp_directory_path() /
                "nonexistent_decision_tree_unlikely_name_42.yaml";
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Reasoner_TreeLoadFailed);
}

// ===========================================================================
// LoadFromYAML_NumericBranchKeys  —  ">10", "<5", ">=3" stored as strings
// ===========================================================================
TEST(DecisionTreeTest, LoadFromYAML_NumericBranchKeys) {
    auto yaml = R"(
type: branch
field: measurement
branches:
  ">10":
    type: leaf
    label: high
    recommendation: flag
  "<5":
    type: leaf
    label: low
    recommendation: ok
  ">=3":
    type: leaf
    label: medium
    recommendation: check
default:
  type: leaf
  label: unknown
  recommendation: review
)";

    auto path = TempYAMLPath(yaml);
    auto result = DecisionTree::LoadFromYAML(path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto& root = static_cast<const BranchNode&>(result.value()->Root());
    auto& branches = root.Branches();

    EXPECT_EQ(branches.size(), 3);
    EXPECT_NE(branches.find(">10"), branches.end());
    EXPECT_NE(branches.find("<5"), branches.end());
    EXPECT_NE(branches.find(">=3"), branches.end());

    auto verify_leaf = [&](const std::string& key,
                           const std::string& expected_label) {
        auto it = branches.find(key);
        ASSERT_NE(it, branches.end());
        ASSERT_EQ(it->second->GetKind(), IDecisionNode::Kind::Leaf);
        EXPECT_EQ(
            static_cast<const LeafNode&>(*it->second).Label(),
            expected_label);
    };

    verify_leaf(">10", "high");
    verify_leaf("<5", "low");
    verify_leaf(">=3", "medium");

    std::filesystem::remove(path);
}

}  // namespace
}  // namespace sai::reasoner
