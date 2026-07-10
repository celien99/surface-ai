#include <sai/plugin/plugin.h>

namespace sai::test::fixture_plugin {

namespace {

int g_initialize_calls = 0;
int g_start_calls = 0;
int g_stop_calls = 0;

// Kept in sync with tests/plugin/fixture_plugin/plugin.yaml by hand — this
// batch does not require GetManifest() to re-read the YAML file, only that
// it reports the same manifest content the plugin was declared with (see
// 1.3-core-plugin-system.md §4: GetManifest exposes the already-validated
// manifest, it does not re-derive it).
auto BuildManifest() -> PluginManifest {
    PluginManifest manifest;
    manifest.name = "sai.test.fixture-plugin";
    manifest.library_path = "fixture_plugin";
    manifest.version = SemVer{.major = 1, .minor = 0, .patch = 0};
    manifest.license_token = "fixture-plugin-license-token";
    return manifest;
}

// The string passed to SAI_DECLARE_TYPE_ID below must match manifest.name
// (and plugin.yaml's `name` field) character-for-character, per
// 1.3-core-plugin-system.md §3: both are hashed with the same
// detail::Fnv1aHash to produce this plugin's TypeId, and PluginManager
// computes Fnv1aHash(manifest.name) rather than calling TypeId() when
// registering/resolving — the two must agree or cross-plugin Resolve() calls
// made via TypeId() would silently target the wrong entry.
class FixturePlugin final : public IPlugin {
public:
    SAI_DECLARE_TYPE_ID(sai.test.fixture-plugin)

    auto OnInitialize(Context& /*context*/) -> Result<void> override {
        ++g_initialize_calls;
        return {};
    }

    auto OnStart(Context& /*context*/) -> Result<void> override {
        ++g_start_calls;
        return {};
    }

    auto OnStop(Context& /*context*/) -> Result<void> override {
        ++g_stop_calls;
        return {};
    }

    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override {
        static const PluginManifest kManifest = BuildManifest();
        return kManifest;
    }
};

}  // namespace

}  // namespace sai::test::fixture_plugin

// The test process loads this library into its own address space via a
// genuine, independent dlopen call (in addition to whatever dlopen
// PluginManager itself performs) purely to observe these counters — POSIX
// guarantees a second dlopen of the same file just bumps a refcount and
// returns a handle into the *same* already-loaded image, so these accessors
// read the exact same g_*_calls that PluginManager's loaded instance wrote
// to. Linking fixture_plugin.cpp directly into the test binary instead would
// give the test its own, unrelated copy of these statics and could never
// observe the dlopen'd instance's calls.
extern "C" auto CreatePlugin() -> sai::IPlugin* {
    return new sai::test::fixture_plugin::FixturePlugin();
}

extern "C" auto DestroyPlugin(sai::IPlugin* plugin) -> void { delete plugin; }

extern "C" auto FixturePlugin_InitializeCallCount() -> int {
    return sai::test::fixture_plugin::g_initialize_calls;
}

extern "C" auto FixturePlugin_StartCallCount() -> int {
    return sai::test::fixture_plugin::g_start_calls;
}

extern "C" auto FixturePlugin_StopCallCount() -> int {
    return sai::test::fixture_plugin::g_stop_calls;
}

extern "C" auto FixturePlugin_ResetCallCounts() -> void {
    sai::test::fixture_plugin::g_initialize_calls = 0;
    sai::test::fixture_plugin::g_start_calls = 0;
    sai::test::fixture_plugin::g_stop_calls = 0;
}
