#include <gtest/gtest.h>
#include <sai/pipeline/stage_node.h>
#include <sai/core/context.h>

namespace sai::pipeline {
namespace {

using sai::image::RawImage;
using sai::image::SurfaceImage;
using sai::detection::DetectionResult;

class MockStage : public IStageNode {
public:
    explicit MockStage(std::string id, StageType type)
        : id_(std::move(id)), type_(type), last_input_(DetectionResult{}) {}

    auto GetType() const noexcept -> StageType override { return type_; }
    auto GetId() const -> std::string_view override { return id_; }

    auto OnInitialize(sai::Context&) -> Result<void> override {
        init_called_ = true;
        return {};
    }
    auto OnStart(sai::Context&) -> Result<void> override {
        start_called_ = true;
        return {};
    }
    auto OnStop(sai::Context&) -> Result<void> override {
        stop_called_ = true;
        return {};
    }
    auto Process(StageInput input) -> Result<StageOutput> override {
        process_called_ = true;
        last_input_ = std::move(input);
        // Echo input as output (passthrough)
        return std::visit([](auto&& val) -> StageOutput {
            return StageOutput(std::move(val));
        }, last_input_);
    }

    bool init_called_ = false;
    bool start_called_ = false;
    bool stop_called_ = false;
    bool process_called_ = false;
    StageInput last_input_;
    std::string id_;
    StageType type_;
};

TEST(IStageNodeTest, LifecycleSequence) {
    MockStage stage("test_stage", StageType::Capture);
    sai::Context ctx;

    EXPECT_FALSE(stage.init_called_);
    ASSERT_TRUE(stage.OnInitialize(ctx).has_value());
    EXPECT_TRUE(stage.init_called_);

    EXPECT_FALSE(stage.start_called_);
    ASSERT_TRUE(stage.OnStart(ctx).has_value());
    EXPECT_TRUE(stage.start_called_);

    EXPECT_FALSE(stage.stop_called_);
    ASSERT_TRUE(stage.OnStop(ctx).has_value());
    EXPECT_TRUE(stage.stop_called_);
}

TEST(IStageNodeTest, ProcessPassthrough) {
    MockStage stage("test", StageType::Capture);
    sai::Context ctx;
    ASSERT_TRUE(stage.OnInitialize(ctx).has_value());
    ASSERT_TRUE(stage.OnStart(ctx).has_value());

    RawImage mock_image = RawImage::FromOwnedBuffer(
        std::vector<std::uint8_t>{1, 2, 3}, sai::image::ImageMeta{});

    StageInput input(std::move(mock_image));
    auto result = stage.Process(std::move(input));
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(stage.process_called_);
    EXPECT_TRUE(std::holds_alternative<RawImage>(result.value()));
}

TEST(IStageNodeTest, GetTypeAndId) {
    MockStage stage("my_id", StageType::Reason);
    EXPECT_EQ(stage.GetType(), StageType::Reason);
    EXPECT_EQ(stage.GetId(), "my_id");
}

} // namespace
} // namespace sai::pipeline
