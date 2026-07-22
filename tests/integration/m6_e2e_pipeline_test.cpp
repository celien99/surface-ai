#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <sai/core/context.h>
#include <sai/pipeline/pipeline.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace {

using namespace sai::pipeline;

std::filesystem::path WritePipelineYaml() {
    const char* yaml = R"(
pipeline:
  name: "e2e_test"
  version: "1.0"
  backpressure:
    default: block
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: preprocess
      type: Preprocess
      depends_on: [capture]
    - id: inference
      type: Inference
      depends_on: [preprocess]
    - id: detect
      type: Detect
      depends_on: [inference]
    - id: rule_eval
      type: RuleEval
      depends_on: [detect]
    - id: reason
      type: Reason
      depends_on: [rule_eval]
    - id: export
      type: Export
      depends_on: [reason]
)";
    auto path = std::filesystem::temp_directory_path() / "m6_e2e_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << yaml;
    return path;
}

TEST(M6E2EPipelineTest, LoadPipelineFromYaml) {
    auto yaml_path = WritePipelineYaml();
    sai::Context ctx;

    auto pipeline = Pipeline::LoadFromYAML(yaml_path, ctx);
    ASSERT_TRUE(pipeline.has_value())
        << "LoadFromYAML failed: " << pipeline.error().message;

    auto metrics = (*pipeline)->Metrics();
    ASSERT_EQ(metrics.size(), 7);

    // Verify stage ids and types
    std::map<std::string, StageType> expected = {
        {"capture", StageType::Capture},
        {"preprocess", StageType::Preprocess},
        {"inference", StageType::Inference},
        {"detect", StageType::Detect},
        {"rule_eval", StageType::RuleEval},
        {"reason", StageType::Reason},
        {"export", StageType::Export},
    };

    for (auto& m : metrics) {
        auto it = expected.find(m.stage_id);
        ASSERT_NE(it, expected.end())
            << "Unexpected stage: " << m.stage_id;
        EXPECT_EQ(m.type, it->second)
            << "Wrong type for stage: " << m.stage_id;
    }
}

TEST(M6E2EPipelineTest, RejectsSubmissionBeforeStart) {
    auto yaml_path = WritePipelineYaml();
    sai::Context ctx;
    auto pipeline = Pipeline::LoadFromYAML(yaml_path, ctx);
    ASSERT_TRUE(pipeline.has_value());

    auto image = sai::image::RawImage::FromOwnedBuffer(
        std::vector<std::uint8_t>{1},
        sai::image::ImageMeta{
            .width = 1,
            .height = 1,
            .channels = 1,
            .pixel_format = sai::image::PixelFormat::Mono8,
        });
    auto result = (*pipeline)->Submit(std::move(image));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Pipeline_InvalidState);
}

TEST(M6E2EPipelineTest, InvalidYamlReturnsError) {
    const char* bad_yaml = R"(
pipeline:
  name: "bad"
  stages:
    - id: a
      type: Capture
      depends_on: []
    - id: b
      type: Capture
      depends_on: [c]
)";
    auto path = std::filesystem::temp_directory_path() / "bad_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << bad_yaml;
    ofs.close();

    sai::Context ctx;
    auto pipeline = Pipeline::LoadFromYAML(path, ctx);
    EXPECT_FALSE(pipeline.has_value());
    EXPECT_EQ(pipeline.error().code, sai::ErrorCode::Pipeline_InvalidConfig);
}

TEST(M6E2EPipelineTest, CyclicDependencyRejected) {
    const char* cyclic_yaml = R"(
pipeline:
  name: "cyclic"
  version: "1.0"
  stages:
    - id: a
      type: Capture
      depends_on: [c]
    - id: b
      type: Capture
      depends_on: [a]
    - id: c
      type: Capture
      depends_on: [b]
)";
    auto path = std::filesystem::temp_directory_path() / "cyclic_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << cyclic_yaml;
    ofs.close();

    sai::Context ctx;
    auto pipeline = Pipeline::LoadFromYAML(path, ctx);
    EXPECT_FALSE(pipeline.has_value());
    EXPECT_EQ(pipeline.error().code, sai::ErrorCode::Pipeline_InvalidConfig);
}

TEST(M6E2EPipelineTest, StageTypeFromStringConversion) {
    EXPECT_EQ(StageTypeFromString("Capture"), StageType::Capture);
    EXPECT_EQ(StageTypeFromString("Preprocess"), StageType::Preprocess);
    EXPECT_EQ(StageTypeFromString("Inference"), StageType::Inference);
    EXPECT_EQ(StageTypeFromString("Detect"), StageType::Detect);
    EXPECT_EQ(StageTypeFromString("RuleEval"), StageType::RuleEval);
    EXPECT_EQ(StageTypeFromString("Reason"), StageType::Reason);
    EXPECT_EQ(StageTypeFromString("Export"), StageType::Export);
    EXPECT_EQ(StageTypeFromString("Custom"), StageType::Custom);
    EXPECT_THROW(StageTypeFromString("Nonexistent"), std::invalid_argument);
}

}  // namespace
