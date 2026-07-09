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

class GreeterService final : public sai::IService {
public:
    SAI_DECLARE_TYPE_ID(sai::test::context_lifecycle_test::GreeterService)

    [[nodiscard]] auto Greet() const -> std::string { return "hello"; }
};

// Registers a GreeterService while assembling, so a later module can resolve it.
class ServiceProvidingModule final : public sai::IModule {
public:
    auto OnInitialize(sai::Context& context) -> sai::Result<void> override {
        return context.Register<GreeterService>(std::make_shared<GreeterService>());
    }

    auto OnStart(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }
    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }
};

// Resolves the GreeterService during OnStart and records whether resolution
// succeeded and what the resolved instance actually returns.
class ServiceConsumingModule final : public sai::IModule {
public:
    ServiceConsumingModule(bool& resolved_ok, std::string& resolved_greeting)
        : resolved_ok_(resolved_ok), resolved_greeting_(resolved_greeting) {}

    auto OnInitialize(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

    auto OnStart(sai::Context& context) -> sai::Result<void> override {
        auto resolved = context.Resolve<GreeterService>();
        resolved_ok_ = resolved.has_value();
        if (!resolved_ok_) {
            return {};
        }
        resolved_greeting_ = resolved.value()->Greet();
        return {};
    }

    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

private:
    bool& resolved_ok_;
    std::string& resolved_greeting_;
};

// Logs its name on OnStop like RecordingModule, but always fails with a fixed
// error code, so tests can verify Stop() keeps going after a failure.
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

TEST(ContextLifecycleTest, ResolveReturnsTheExactRegisteredInstance) {
    sai::Context context;
    auto registered = std::make_shared<ProbeService>();

    ASSERT_TRUE(context.Register<ProbeService>(registered).has_value());

    auto resolved = context.Resolve<ProbeService>();

    ASSERT_TRUE(resolved.has_value());
    // Identity check: same shared_ptr-managed object, not merely equal content.
    EXPECT_EQ(resolved.value(), registered);
    EXPECT_EQ(resolved.value().get(), registered.get());
}

TEST(ContextLifecycleTest, ResolveOfUnregisteredServiceReportsTypeNotFound) {
    sai::Context context;

    auto resolved = context.Resolve<ProbeService>();

    ASSERT_FALSE(resolved.has_value());
    EXPECT_EQ(resolved.error().code, sai::ErrorCode::Core_TypeNotFound);
}

TEST(ContextLifecycleTest, ModuleCanResolveServiceRegisteredByEarlierModuleDuringOnStart) {
    sai::Context context;
    bool resolved_ok = false;
    std::string resolved_greeting;

    ASSERT_TRUE(
        context.RegisterModule(std::make_unique<ServiceProvidingModule>()).has_value());
    ASSERT_TRUE(context
        .RegisterModule(
            std::make_unique<ServiceConsumingModule>(resolved_ok, resolved_greeting))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());

    ASSERT_TRUE(resolved_ok);
    EXPECT_EQ(resolved_greeting, "hello");
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

    // OnStop runs in reverse registration order, so "second" fails first and
    // "first" fails second; Stop() must still call both and keep the first
    // error encountered during that reverse walk.
    auto result = context.Stop();

    EXPECT_EQ(stop_log, (std::vector<std::string>{"second", "first"}));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Core_TypeAlreadyRegistered);
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Stopped);
}
