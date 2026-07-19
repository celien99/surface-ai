#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sai/core/context.h>
#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/plugin/plugin.h>

namespace sai {

// PluginManager — dynamic .so plugin loader + capability/license/version validation.
//
// Batch T3 refactor: merged CapabilityManager, LicenseManager, VersionManager,
// and ModuleManager into PluginManager.  6 classes → 2 (PluginManager + PluginManifest).
//
// ModuleManager deleted — its RegisterBuiltin was a trivial wrapper around
// Context::RegisterModule; callers can call ctx.RegisterModule() directly.
class PluginManager {
public:
    explicit PluginManager(Context& context) noexcept;
    ~PluginManager();

    PluginManager(const PluginManager&) = delete;
    PluginManager& operator=(const PluginManager&) = delete;
    PluginManager(PluginManager&&) = delete;
    PluginManager& operator=(PluginManager&&) = delete;

    // Register a capability tag known to this deployment.
    // Duplicate registration returns Core_TypeAlreadyRegistered.
    auto RegisterKnownCapability(std::string capability) -> Result<void>;

    // Scans plugin.yaml files under `plugin_dir` and builds the manifest index.
    auto DiscoverManifests(const std::filesystem::path& plugin_dir) -> Result<void>;

    // Loads a plugin by name (with recursive dependency resolution).
    auto Load(const std::string& plugin_name) -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<std::shared_ptr<IPlugin>>;

    // Application-wide shutdown: calls OnStop on every loaded plugin in
    // reverse load order.
    auto Shutdown() -> Result<void>;

    // Inline validators (formerly VersionManager / LicenseManager / CapabilityManager)
    [[nodiscard]] static auto CheckVersion(const VersionRange& required,
                                            const SemVer& actual) noexcept -> Result<void>;
    [[nodiscard]] static auto ValidateLicense(std::string_view token) noexcept -> Result<void>;
    [[nodiscard]] auto ValidateCapabilities(
        const std::vector<std::string>& declared) const -> Result<void>;

private:
    auto LoadSingle(const PluginManifest& manifest) -> Result<void>;
    auto EnsureLoaded(const std::string& plugin_name,
                       std::unordered_set<std::string>& visiting) -> Result<void>;
    auto ResolveEach(const std::vector<PluginDependency>& deps,
                      std::unordered_set<std::string>& visiting) -> Result<void>;
    auto ShutdownFrom(std::size_t remaining, Result<void> first_error) -> Result<void>;

    [[nodiscard]] auto FindManifest(const std::string& plugin_name) const
        -> Result<PluginManifest>;

    Context& context_;
    Registry<IPlugin> plugin_registry_;

    // Capability manager state (inline)
    mutable std::shared_mutex cap_mutex_;
    std::unordered_set<std::string> known_capabilities_;

    std::unordered_map<std::string, PluginManifest> manifest_index_;
    std::unordered_map<std::string, void*> library_handles_;
    std::vector<TypeId> load_order_;
};

}  // namespace sai
