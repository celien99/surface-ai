#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sai/core/context.h>
#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/plugin/capability_manager.h>
#include <sai/plugin/license_manager.h>
#include <sai/plugin/plugin.h>
#include <sai/plugin/version_manager.h>

namespace sai {

// Runtime manager for dynamically loaded .so plugins. Holds a
// Registry<IPlugin> (instantiation of the 1.2 template, not redefined) so
// plugins can Resolve each other by TypeId, and so the rest of the
// framework can resolve already-loaded plugins the same way. See
// 1.3-core-plugin-system.md §3/§4/§5 — Context never holds a pointer to any
// plugin instance; PluginManager drives IPlugin's lifecycle hooks itself.
class PluginManager {
public:
    explicit PluginManager(Context& context) noexcept;
    ~PluginManager();

    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;
    PluginManager(PluginManager&&) = delete;
    PluginManager& operator=(PluginManager&&) = delete;

    // Scans plugin.yaml files under `plugin_dir` (any depth) and builds a
    // name -> PluginManifest index. Reads manifest text only — no dlopen,
    // no version/capability/license validation (a pure discovery step, see
    // §3 Design).
    auto DiscoverManifests(const std::filesystem::path& plugin_dir) -> Result<void>;

    // Loads a plugin by name: recursively ensures its dependencies are
    // loaded first (see §5 Workflow), then loads the plugin itself. If a
    // plugin with the same TypeId is already in plugin_registry_, returns
    // success immediately (idempotent).
    auto Load(const std::string& plugin_name) -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<std::shared_ptr<IPlugin>>;

    // Application-wide stop entry point: calls OnStop on every loaded
    // plugin in plugin_registry_, in the reverse of load_order_ (same
    // rationale as Context::Stop()'s reverse shutdown, see §3 Design). The
    // application entry point must call this before context.Stop(). Same
    // error-handling policy as Context::Stop(): a single plugin's OnStop
    // failure does not block the remaining plugins; the return value is the
    // first error encountered during the walk (or success if all succeed).
    auto Shutdown() -> Result<void>;

private:
    auto LoadSingle(const PluginManifest& manifest) -> Result<void>;
    auto EnsureLoaded(const std::string& plugin_name,
                       std::unordered_set<std::string>& visiting) -> Result<void>;
    auto ResolveEach(const std::vector<PluginDependency>& deps,
                      std::unordered_set<std::string>& visiting) -> Result<void>;

    // Tail-to-head recursion over load_order_, same shape as ResolveEach's
    // head/tail recursion, walking the opposite direction to express
    // "reverse of load order". `remaining` carries the not-yet-processed
    // prefix length, `first_error` is the first failure seen so far
    // (initially success); the base case is remaining == 0.
    auto ShutdownFrom(std::size_t remaining, Result<void> first_error) -> Result<void>;

    [[nodiscard]] auto FindManifest(const std::string& plugin_name) const
        -> Result<PluginManifest>;

    Context& context_;
    VersionManager version_manager_;
    LicenseManager license_manager_;
    CapabilityManager capability_manager_;
    Registry<IPlugin> plugin_registry_;
    std::unordered_map<std::string, PluginManifest> manifest_index_;
    std::unordered_map<std::string, void*> library_handles_;
    // Order in which plugins were successfully Register()'d into
    // plugin_registry_ (appended inside LoadSingle); Registry<IPlugin> is
    // backed by an unordered_map internally and does not preserve insertion
    // order, so Shutdown() needs this separate, order-preserving record to
    // call OnStop in the reverse of load order.
    std::vector<TypeId> load_order_;
};

}  // namespace sai
