#include <sai/core/context.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

class RecordingModule final : public sai::IModule {
public:
    RecordingModule(std::string name, std::vector<std::string>& initialize_log,
                     std::vector<std::string>& stop_log)
        : name_(std::move(name)), initialize_log_(initialize_log), stop_log_(stop_log) {}

    auto OnInitialize(sai::Context& /*context*/) -> sai::Result<void> override {
        initialize_log_.push_back(name_);
        return {};
    }

    auto OnStart(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override {
        stop_log_.push_back(name_);
        return {};
    }

private:
    std::string name_;
    std::vector<std::string>& initialize_log_;
    std::vector<std::string>& stop_log_;
};

// Logs its name on OnStop, but always fails with a fixed error code.
class FailOnStopModule final : public sai::IModule {
public:
    FailOnStopModule(std::string name, std::vector<std::string>& stop_log,
                      sai::ErrorCode error_code)
        : name_(std::move(name)), stop_log_(stop_log), error_code_(error_code) {}

    auto OnInitialize(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }
    auto OnStart(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override {
        stop_log_.push_back(name_);
        return tl::make_unexpected(sai::ErrorInfo{
            error_code_,
            "forced stop failure for test",
            std::source_location::current(),
        });
    }

private:
    std::string name_;
    std::vector<std::string>& stop_log_;
    sai::ErrorCode error_code_;
};

}  // namespace

// IService / Context::Register / Context::Resolve removed in Batch T3.
// Dependencies are injected via setter methods instead of DI container.

TEST(ContextLifecycleTest, FullAssemblyWalksStatesInOrder) {
    sai::Context context;
    std::vector<std::string> initialize_log;
    std::vector<std::string> stop_log;

    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Created);

    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("only", initialize_log, stop_log))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Initialized);

    ASSERT_TRUE(context.Start().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Running);

    ASSERT_TRUE(context.Stop().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Stopped);
}

TEST(ContextLifecycleTest, OnStopRunsInReverseRegistrationOrder) {
    sai::Context context;
    std::vector<std::string> initialize_log;
    std::vector<std::string> stop_log;

    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("first", initialize_log, stop_log))
        .has_value());
    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("second", initialize_log, stop_log))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());
    ASSERT_TRUE(context.Stop().has_value());

    EXPECT_EQ(initialize_log, (std::vector<std::string>{"first", "second"}));
    EXPECT_EQ(stop_log, (std::vector<std::string>{"second", "first"}));
}

TEST(ContextLifecycleTest, StopContinuesCallingRemainingModulesAfterAnEarlierFailure) {
    sai::Context context;
    std::vector<std::string> stop_log;

    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<FailOnStopModule>(
            "first", stop_log, sai::ErrorCode::Core_ConstructionFailed))
        .has_value());
    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<FailOnStopModule>(
            "second", stop_log, sai::ErrorCode::Core_TypeAlreadyRegistered))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());

    auto result = context.Stop();

    EXPECT_EQ(stop_log, (std::vector<std::string>{"second", "first"}));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Core_TypeAlreadyRegistered);
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Stopped);
}
