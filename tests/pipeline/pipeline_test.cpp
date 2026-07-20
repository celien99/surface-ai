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
        : id_(std::move(id)), type_(type) {}

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
        last_type_index_ = input.TypeIndex();
        // Echo input as output (passthrough via move)
        return input;
    }

    bool init_called_ = false;
    bool start_called_ = false;
    bool stop_called_ = false;
    bool process_called_ = false;
    int last_type_index_ = -1;
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

    StageInput input = StageInput::Make(std::move(mock_image));
    auto result = stage.Process(std::move(input));
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(stage.process_called_);
    // Passthrough echoed RawImage input
    EXPECT_NE(result.value().GetIf<RawImage>(), nullptr);
}

TEST(IStageNodeTest, GetTypeAndId) {
    MockStage stage("my_id", StageType::Reason);
    EXPECT_EQ(stage.GetType(), StageType::Reason);
    EXPECT_EQ(stage.GetId(), "my_id");
}

class FailingStage : public IStageNode {
public:
    FailingStage(std::string id, bool fail_on_process = true)
        : id_(std::move(id)), fail_(fail_on_process) {}

    auto GetType() const noexcept -> StageType override { return StageType::Custom; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }
    auto Process(StageInput) -> Result<StageOutput> override {
        if (fail_) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Pipeline_StageInitFailed,
                "simulated stage failure"});
        }
        return StageOutput::Make(RawImage::FromOwnedBuffer(
            std::vector<uint8_t>{42}, sai::image::ImageMeta{}));
    }

    void SetFail(bool f) { fail_ = f; }

private:
    std::string id_;
    bool fail_;
};

TEST(PipelineFailureTest, StageFailureDoesNotCrash) {
    // Verify that a failing Process() returns an error, not an exception
    FailingStage stage("failing", true);
    sai::Context ctx;
    (void)stage.OnInitialize(ctx);

    RawImage mock = RawImage::FromOwnedBuffer(
        std::vector<uint8_t>{1, 2, 3}, sai::image::ImageMeta{});
    StageInput input = StageInput::Make(std::move(mock));
    auto result = stage.Process(std::move(input));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Pipeline_StageInitFailed);
}

TEST(PipelineFailureTest, StageRecoversAfterFailure) {
    // A stage that fails once then succeeds on the next call
    FailingStage stage("recover", true);
    sai::Context ctx;
    (void)stage.OnInitialize(ctx);

    // First call fails
    {
        RawImage mock = RawImage::FromOwnedBuffer(
            std::vector<uint8_t>{1}, sai::image::ImageMeta{});
        EXPECT_FALSE(stage.Process(StageInput::Make(std::move(mock))).has_value());
    }

    // Second call succeeds (stage recovered)
    stage.SetFail(false);
    {
        RawImage mock = RawImage::FromOwnedBuffer(
            std::vector<uint8_t>{1}, sai::image::ImageMeta{});
        auto result = stage.Process(StageInput::Make(std::move(mock)));
        EXPECT_TRUE(result.has_value());
    }
}

} // namespace
} // namespace sai::pipeline
