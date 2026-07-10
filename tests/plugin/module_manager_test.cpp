#include <sai/plugin/module_manager.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

class RecordingModule final : public sai::IModule {
public:
    RecordingModule(std::string name, std::vector<std::string>& initialize_log)
        : name_(std::move(name)), initialize_log_(initialize_log) {}

    auto OnInitialize(sai::Context& /*context*/) -> sai::Result<void> override {
        initialize_log_.push_back(name_);
        return {};
    }

    auto OnStart(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }
    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

private:
    std::string name_;
    std::vector<std::string>& initialize_log_;
};

}  // namespace

TEST(ModuleManagerTest, RegisterBuiltinForwardsToContextRegisterModule) {
    sai::Context context;
    sai::ModuleManager module_manager(context);
    std::vector<std::string> initialize_log;

    auto result =
        module_manager.RegisterBuiltin(std::make_unique<RecordingModule>("only", initialize_log));

    ASSERT_TRUE(result.has_value());

    // The module landed in the same Context that would have received it via
    // a direct Context::RegisterModule call: driving Initialize() runs its
    // OnInitialize exactly like the context_lifecycle_test.cpp baseline does.
    ASSERT_TRUE(context.Initialize().has_value());
    EXPECT_EQ(initialize_log, (std::vector<std::string>{"only"}));
}

TEST(ModuleManagerTest, RegisterBuiltinAfterAssemblyIsRejectedLikeDirectRegisterModule) {
    sai::Context context;
    sai::ModuleManager module_manager(context);

    ASSERT_TRUE(context.Initialize().has_value());

    std::vector<std::string> initialize_log;
    auto result =
        module_manager.RegisterBuiltin(std::make_unique<RecordingModule>("late", initialize_log));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Lifecycle_RegisterAfterAssembly);
}
