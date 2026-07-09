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

class ProbeService final : public sai::IService {
public:
    SAI_DECLARE_TYPE_ID(sai::test::context_lifecycle_test::ProbeService)
};

}  // namespace

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

TEST(ContextLifecycleTest, RegisterAfterAssemblyIsRejectedWhileRunning) {
    sai::Context context;
    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());
    ASSERT_EQ(context.CurrentState(), sai::LifecycleState::Running);

    auto result = context.Register<ProbeService>(std::make_shared<ProbeService>());

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Lifecycle_RegisterAfterAssembly);
}
