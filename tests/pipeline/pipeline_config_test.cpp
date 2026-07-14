#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sai/pipeline/pipeline_config.h>
#include <sai/core/error.h>
#include <yaml-cpp/yaml.h>

namespace sai::pipeline {
namespace {

// Helper: write a YAML string to a temp file
std::filesystem::path WriteTempYaml(const std::string& content) {
    auto path = std::filesystem::temp_directory_path() / "test_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

TEST(PipelineConfigTest, ParseMinimalPipeline) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto node = root["pipeline"];
    ASSERT_TRUE(node.IsDefined());

    auto name = node["name"].as<std::string>();
    EXPECT_EQ(name, "test");
    auto version = node["version"].as<std::string>();
    EXPECT_EQ(version, "1.0");
    auto stages = node["stages"];
    ASSERT_EQ(stages.size(), 2);
    EXPECT_EQ(stages[0]["id"].as<std::string>(), "capture");
    EXPECT_EQ(stages[0]["type"].as<std::string>(), "Capture");
}

TEST(PipelineConfigTest, BackpressureDefaults) {
    BackpressureConfig bp;
    EXPECT_EQ(bp.default_policy, BackpressurePolicy::Block);
    EXPECT_TRUE(bp.stage_overrides.empty());
}

TEST(PipelineConfigTest, StageTypeFromString) {
    EXPECT_EQ(StageTypeFromString("Capture"), StageType::Capture);
    EXPECT_EQ(StageTypeFromString("Preprocess"), StageType::Preprocess);
    EXPECT_EQ(StageTypeFromString("Inference"), StageType::Inference);
    EXPECT_EQ(StageTypeFromString("Detect"), StageType::Detect);
    EXPECT_EQ(StageTypeFromString("RuleEval"), StageType::RuleEval);
    EXPECT_EQ(StageTypeFromString("Reason"), StageType::Reason);
    EXPECT_EQ(StageTypeFromString("Export"), StageType::Export);
    EXPECT_EQ(StageTypeFromString("Custom"), StageType::Custom);
}

TEST(PipelineConfigTest, StageTypeFromStringInvalid) {
    EXPECT_THROW(StageTypeFromString("Nonexistent"), std::invalid_argument);
}

TEST(PipelineConfigTest, BackpressurePolicyFromString) {
    EXPECT_EQ(BackpressurePolicyFromString("block"), BackpressurePolicy::Block);
    EXPECT_EQ(BackpressurePolicyFromString("drop_oldest"), BackpressurePolicy::DropOldest);
    EXPECT_EQ(BackpressurePolicyFromString("degrade"), BackpressurePolicy::Degrade);
}

TEST(PipelineConfigTest, BackpressurePolicyFromStringInvalid) {
    EXPECT_THROW(BackpressurePolicyFromString("invalid"), std::invalid_argument);
}

TEST(PipelineConfigTest, ParseValidConfig) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto pipe = root["pipeline"];
    ASSERT_TRUE(pipe.IsDefined());
    EXPECT_EQ(pipe["name"].as<std::string>(), "test");
    EXPECT_EQ(pipe["version"].as<std::string>(), "1.0");
}

TEST(PipelineConfigTest, ParseWithBackpressureOverride) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  backpressure:
    default: block
    stage_overrides:
      capture: drop_oldest
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto bp = root["pipeline"]["backpressure"];
    EXPECT_EQ(bp["default"].as<std::string>(), "block");
    EXPECT_EQ(bp["stage_overrides"]["capture"].as<std::string>(), "drop_oldest");
}

} // namespace
} // namespace sai::pipeline
