#include <sai/plugin/plugin_manager.h>

#include <dlfcn.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace {

// Populated by CMake via target_compile_definitions (see
// tests/plugin/CMakeLists.txt): the absolute, platform-correct path to the
// real fixture_plugin SHARED library artifact (.dylib here, .so on the
// target platform) and the directory holding its checked-in plugin.yaml.
constexpr const char* kFixtureLibraryFile = FIXTURE_PLUGIN_LIBRARY_FILE;
constexpr const char* kFixtureSourceDir = FIXTURE_PLUGIN_SOURCE_DIR;
constexpr const char* kFixturePluginName = "sai.test.fixture-plugin";

auto MakeTempDir(const std::string& suffix) -> std::filesystem::path {
    auto dir = std::filesystem::temp_directory_path() /
               ("sai_plugin_manager_test_" + suffix);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// Writes a plugin.yaml under `dir` whose library_path is `library_path`
// (verbatim — an absolute path here overrides PluginManager's
// parent-directory join, see plugin_manager.cpp's DiscoverManifests) and
// whose dependencies list is `dependency_name` (empty string means no
// dependency declared).
void WriteManifest(const std::filesystem::path& dir, const std::string& name,
                    const std::string& library_path, const std::string& dependency_name) {
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "plugin.yaml");
    out << "name: " << name << "\n";
    out << "library_path: " << library_path << "\n";
    out << "version:\n  major: 1\n  minor: 0\n  patch: 0\n";
    out << "capabilities: []\n";
    if (dependency_name.empty()) {
        out << "dependencies: []\n";
    } else {
        out << "dependencies:\n";
        out << "  - plugin_name: " << dependency_name << "\n";
        out << "    required_version:\n";
        out << "      min_inclusive: {major: 0, minor: 0, patch: 0}\n";
        out << "      max_exclusive: {major: 99, minor: 0, patch: 0}\n";
    }
    out << "license_token: some-token\n";
}

// Opens the real fixture library directly (independent of any handle
// PluginManager holds) purely to reach its call-count observers. dlopen on
// POSIX resolves by canonical path: opening the exact same absolute path
// PluginManager itself dlopens returns a handle refcounted against the same
// already-loaded image, so these accessors read the exact statics the
// dlopen'd FixturePlugin instance wrote to — not a separate copy. This is
// the only way to observe the fixture's lifecycle calls without linking
// fixture_plugin.cpp into this test binary, which would defeat the point of
// testing a real dlopen (see tests/plugin/CMakeLists.txt).
class FixtureLibraryProbe {
public:
    FixtureLibraryProbe() {
        handle_ = dlopen(kFixtureLibraryFile, RTLD_NOW | RTLD_LOCAL);
    }

    ~FixtureLibraryProbe() {
        if (handle_ != nullptr) {
            dlclose(handle_);
        }
    }

    FixtureLibraryProbe(const FixtureLibraryProbe&) = delete;
    FixtureLibraryProbe& operator=(const FixtureLibraryProbe&) = delete;

    [[nodiscard]] auto IsOpen() const noexcept -> bool { return handle_ != nullptr; }

    auto Reset() const -> void { Symbol<void(*)()>("FixturePlugin_ResetCallCounts")(); }
    [[nodiscard]] auto InitializeCallCount() const -> int {
        return Symbol<int(*)()>("FixturePlugin_InitializeCallCount")();
    }
    [[nodiscard]] auto StartCallCount() const -> int {
        return Symbol<int(*)()>("FixturePlugin_StartCallCount")();
    }
    [[nodiscard]] auto StopCallCount() const -> int {
        return Symbol<int(*)()>("FixturePlugin_StopCallCount")();
    }

private:
    template <typename FnPtr>
    auto Symbol(const char* name) const -> FnPtr {
        return reinterpret_cast<FnPtr>(dlsym(handle_, name));
    }

    void* handle_ = nullptr;
};

class PluginManagerFixtureTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(probe_.IsOpen()) << "failed to open fixture library for observation: "
                                      << kFixtureLibraryFile;
        probe_.Reset();
        working_dir_ = MakeTempDir(::testing::UnitTest::GetInstance()->current_test_info()->name());
        WriteManifest(working_dir_, kFixturePluginName, kFixtureLibraryFile, "");
    }

    void TearDown() override { std::filesystem::remove_all(working_dir_); }

    FixtureLibraryProbe probe_;
    std::filesystem::path working_dir_;
};

}  // namespace

TEST(PluginManagerTest, DiscoverManifestsFindsFixtureManifestWithoutDlopening) {
    sai::Context context;
    sai::PluginManager plugin_manager(context);

    ASSERT_TRUE(plugin_manager.DiscoverManifests(kFixtureSourceDir).has_value());

    // The checked-in fixture manifest's library_path ("fixture_plugin") does
    // not resolve to a real file, so a Load() attempt against it must fail
    // once it reaches dlopen — but only after successfully finding the
    // manifest by name and passing capability/license validation. Failing
    // with Core_ConstructionFailed (the dlopen failure code) rather than
    // Core_TypeNotFound (the "manifest never discovered" code) is the
    // observable proof that DiscoverManifests actually indexed the manifest,
    // since PluginManager exposes no direct accessor for manifest_index_.
    auto load_result = plugin_manager.Load("sai.test.fixture-plugin");

    ASSERT_FALSE(load_result.has_value());
    EXPECT_EQ(load_result.error().code, sai::ErrorCode::Core_ConstructionFailed);
}

TEST_F(PluginManagerFixtureTest, LoadRunsOnInitializeAndOnStartOnTheRealDlopenedInstance) {
    sai::Context context;
    sai::PluginManager plugin_manager(context);

    ASSERT_TRUE(plugin_manager.DiscoverManifests(working_dir_).has_value());
    auto load_result = plugin_manager.Load(kFixturePluginName);

    ASSERT_TRUE(load_result.has_value()) << load_result.error().message;
    EXPECT_EQ(probe_.InitializeCallCount(), 1);
    EXPECT_EQ(probe_.StartCallCount(), 1);
}

TEST_F(PluginManagerFixtureTest, ResolveAfterLoadReturnsTheLoadedInstance) {
    sai::Context context;
    sai::PluginManager plugin_manager(context);

    ASSERT_TRUE(plugin_manager.DiscoverManifests(working_dir_).has_value());
    ASSERT_TRUE(plugin_manager.Load(kFixturePluginName).has_value());

    auto resolved = plugin_manager.Resolve(sai::detail::Fnv1aHash(kFixturePluginName));

    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved.value()->GetManifest().name, kFixturePluginName);
}

TEST_F(PluginManagerFixtureTest, ShutdownRunsOnStopOnTheRealDlopenedInstance) {
    sai::Context context;
    sai::PluginManager plugin_manager(context);

    ASSERT_TRUE(plugin_manager.DiscoverManifests(working_dir_).has_value());
    ASSERT_TRUE(plugin_manager.Load(kFixturePluginName).has_value());

    auto shutdown_result = plugin_manager.Shutdown();

    ASSERT_TRUE(shutdown_result.has_value()) << shutdown_result.error().message;
    EXPECT_EQ(probe_.StopCallCount(), 1);
}

TEST_F(PluginManagerFixtureTest, LoadingAnAlreadyLoadedPluginTwiceIsIdempotent) {
    sai::Context context;
    sai::PluginManager plugin_manager(context);

    ASSERT_TRUE(plugin_manager.DiscoverManifests(working_dir_).has_value());
    ASSERT_TRUE(plugin_manager.Load(kFixturePluginName).has_value());
    ASSERT_TRUE(plugin_manager.Load(kFixturePluginName).has_value());

    EXPECT_EQ(probe_.InitializeCallCount(), 1);
    EXPECT_EQ(probe_.StartCallCount(), 1);
}

TEST(PluginManagerTest, LoadWithDependencyNotInManifestIndexFailsCleanly) {
    auto dir = MakeTempDir("missing_dependency");
    WriteManifest(dir, "plugin-a", "unused-a.so", "plugin-missing");

    sai::Context context;
    sai::PluginManager plugin_manager(context);
    ASSERT_TRUE(plugin_manager.DiscoverManifests(dir).has_value());

    auto load_result = plugin_manager.Load("plugin-a");

    ASSERT_FALSE(load_result.has_value());
    EXPECT_EQ(load_result.error().code, sai::ErrorCode::Core_TypeNotFound);

    std::filesystem::remove_all(dir);
}

TEST(PluginManagerTest, CircularDependencyIsCaughtByTheVisitingSet) {
    auto dir = MakeTempDir("circular_dependency");
    // plugin-a depends on plugin-b, plugin-b depends on plugin-a: genuinely
    // exercises the visiting set (EnsureLoaded("plugin-a") is reached a
    // second time while still mid-recursion resolving its own dependency
    // chain), rather than a synthetic case that happens not to need it.
    WriteManifest(dir / "plugin-a", "plugin-a", "unused-a.so", "plugin-b");
    WriteManifest(dir / "plugin-b", "plugin-b", "unused-b.so", "plugin-a");

    sai::Context context;
    sai::PluginManager plugin_manager(context);
    ASSERT_TRUE(plugin_manager.DiscoverManifests(dir).has_value());

    auto load_result = plugin_manager.Load("plugin-a");

    ASSERT_FALSE(load_result.has_value());
    EXPECT_EQ(load_result.error().code, sai::ErrorCode::Plugin_CircularDependency);

    std::filesystem::remove_all(dir);
}
