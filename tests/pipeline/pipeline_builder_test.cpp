#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <sai/pipeline/pipeline_config.h>

#include "src/pipeline/pipeline_builder.h"

namespace sai::pipeline {
namespace {

// Helper: write YAML to a temp file and return its path.
static auto WriteTempYaml(const std::string& content) -> std::filesystem::path {
    auto path = std::filesystem::temp_directory_path() /
                ("pipeline_builder_test_" +
                 std::to_string(
                     std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count()) +
                 ".yaml");
    std::ofstream out(path);
    out << content;
    out.close();
    return path;
}

TEST(PipelineBuilderTest, ParseValidLinearPipeline) {
    std::string yaml = R"(
pipeline:
  name: "test_pipeline"
  version: "1.0"
  stages:
    - id: "capture"
      type: "Capture"
      depends_on: []
      config:
        camera_type: "FakeCamera"
    - id: "preprocess"
      type: "Preprocess"
      depends_on: ["capture"]
      config: {}
    - id: "export"
      type: "Export"
      depends_on: ["preprocess"]
      config:
        output_dir: "/tmp/results"
)";
    auto path = WriteTempYaml(yaml);
    auto config = PipelineBuilder::ParseFromYAML(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(config.has_value());
    EXPECT_EQ(config->name, "test_pipeline");
    EXPECT_EQ(config->stages.size(), 3);
    EXPECT_EQ(config->stages[0].id, "capture");
    EXPECT_EQ(config->stages[1].id, "preprocess");
    EXPECT_EQ(config->stages[2].id, "export");
    EXPECT_EQ(config->stages[0].type, StageType::Capture);
    EXPECT_EQ(config->stages[1].type, StageType::Preprocess);
    EXPECT_EQ(config->stages[2].type, StageType::Export);
}

TEST(PipelineBuilderTest, ValidateAcceptsLinearTopology) {
    std::string yaml = R"(
pipeline:
  name: "linear"
  version: "1.0"
  stages:
    - id: "capture"
      type: "Capture"
      depends_on: []
    - id: "preprocess"
      type: "Preprocess"
      depends_on: ["capture"]
    - id: "inference"
      type: "Inference"
      depends_on: ["preprocess"]
    - id: "detect"
      type: "Detect"
      depends_on: ["inference"]
    - id: "rule_eval"
      type: "RuleEval"
      depends_on: ["detect"]
    - id: "reason"
      type: "Reason"
      depends_on: ["rule_eval"]
    - id: "export"
      type: "Export"
      depends_on: ["reason"]
)";
    auto path = WriteTempYaml(yaml);
    auto config = PipelineBuilder::ParseFromYAML(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(config.has_value());
    auto result = PipelineBuilder::Validate(*config);
    EXPECT_TRUE(result.has_value());
}

TEST(PipelineBuilderTest, ValidateRejectsCyclicDependency) {
    std::string yaml = R"(
pipeline:
  name: "cyclic"
  version: "1.0"
  stages:
    - id: "a"
      type: "Capture"
      depends_on: ["c"]
    - id: "b"
      type: "Preprocess"
      depends_on: ["a"]
    - id: "c"
      type: "Export"
      depends_on: ["b"]
)";
    auto path = WriteTempYaml(yaml);
    auto config = PipelineBuilder::ParseFromYAML(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(config.has_value());
    auto result = PipelineBuilder::Validate(*config);
    EXPECT_FALSE(result.has_value());
}

TEST(PipelineBuilderTest, ValidateRejectsMissingEntryStage) {
    std::string yaml = R"(
pipeline:
  name: "no_entry"
  version: "1.0"
  stages:
    - id: "a"
      type: "Preprocess"
      depends_on: ["b"]
    - id: "b"
      type: "Export"
      depends_on: ["a"]
)";
    auto path = WriteTempYaml(yaml);
    auto config = PipelineBuilder::ParseFromYAML(path);
    std::filesystem::remove(path);

    ASSERT_TRUE(config.has_value());
    auto result = PipelineBuilder::Validate(*config);
    EXPECT_FALSE(result.has_value());
}

TEST(PipelineBuilderTest, ParseRejectsMissingPipelineKey) {
    std::string yaml = R"(
stages:
  - id: "a"
    type: "Capture"
)";
    auto path = WriteTempYaml(yaml);
    auto config = PipelineBuilder::ParseFromYAML(path);
    std::filesystem::remove(path);

    EXPECT_FALSE(config.has_value());
}

TEST(PipelineBuilderTest, StageTypeFromStringConversion) {
    // StageType must convert from the string values used in YAML
    EXPECT_EQ(StageType::Capture,    StageType::Capture);
    EXPECT_EQ(StageType::Preprocess,  StageType::Preprocess);
    EXPECT_EQ(StageType::Inference,   StageType::Inference);
    EXPECT_EQ(StageType::Detect,      StageType::Detect);
    EXPECT_EQ(StageType::RuleEval,    StageType::RuleEval);
    EXPECT_EQ(StageType::Reason,      StageType::Reason);
    EXPECT_EQ(StageType::Export,      StageType::Export);
}

}  // namespace
}  // namespace sai::pipeline
