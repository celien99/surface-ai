#include <gtest/gtest.h>
#include <sai/pipeline/pipeline.h>
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

TEST(StageDataTest, MovingStageOutputPreservesConcreteTypeAndFrame) {
    auto frame = std::make_shared<FrameContext>();
    frame->frame_id = 17;

    auto output = StageOutput::MakeWithContext(
        frame,
        RawImage::FromOwnedBuffer(
            std::vector<std::uint8_t>{1, 2, 3}, sai::image::ImageMeta{}));
    auto forwarded = StageOutput(std::move(output));
    forwarded.AttachFrame(frame);

    EXPECT_NE(forwarded.GetIf<RawImage>(), nullptr);
    ASSERT_NE(forwarded.Frame(), nullptr);
    EXPECT_EQ(forwarded.Frame()->frame_id, 17U);
}

TEST(FrameCompletionTest, LastFrameReferenceCompletesLifecycle) {
    auto state = std::make_shared<detail::FrameCompletionState>();
    auto frame = std::make_shared<FrameContext>();
    frame->completion = std::make_shared<detail::FrameCompletionToken>(state);

    EXPECT_FALSE(state->WaitUntil(std::chrono::steady_clock::now()));
    frame.reset();
    EXPECT_TRUE(state->WaitUntil(std::chrono::steady_clock::now()));
}

TEST(FrameContextTest, OwnsMovedSurfaceImageWithoutSnapshotCopy) {
    auto image = SurfaceImage::FromOwnedBuffer(
        std::vector<std::uint8_t>{4, 5, 6},
        sai::image::ImageMeta{
            .width = 1,
            .height = 1,
            .channels = 3,
            .pixel_format = sai::image::PixelFormat::RGB8,
        });
    auto* original = image.Data();

    FrameContext frame;
    frame.image.emplace(std::move(image));

    ASSERT_TRUE(frame.image.has_value());
    EXPECT_EQ(frame.image->Data(), original);
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

TEST(PipelineFailureTest, FailureMessageCarriesFrameRoutingMetadata) {
    PipelineFailure failure{
        .code = ErrorCode::Inference_EngineExecutionFailed,
        .stage_id = "inference",
        .message = "engine unavailable",
        .surface_id = "seat-42",
        .position_id = 3,
    };

    auto message = StageOutput::Make(std::move(failure));
    const auto* received = message.GetIf<PipelineFailure>();
    ASSERT_NE(received, nullptr);
    EXPECT_EQ(received->code, ErrorCode::Inference_EngineExecutionFailed);
    EXPECT_EQ(received->stage_id, "inference");
    EXPECT_EQ(received->surface_id, "seat-42");
    EXPECT_EQ(received->position_id, 3);
}

} // namespace
} // namespace sai::pipeline
