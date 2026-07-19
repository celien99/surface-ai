#include <sai/plugin/plugin_manager.h>

#include <dlfcn.h>

#include <mutex>
#include <source_location>
#include <utility>

namespace sai {

namespace {

auto DlOpen(const std::string& library_path) -> Result<void*> {
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* dl_error = dlerror();
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            std::string("dlopen failed for '") + library_path + "': " +
                (dl_error != nullptr ? dl_error : "unknown error"),
            std::source_location::current(),
        });
    }
    return handle;
}

auto CreateInstance(void* handle, const PluginManifest& manifest)
    -> Result<std::shared_ptr<IPlugin>> {
    dlerror();
    auto* create_fn = reinterpret_cast<CreatePluginFn>(dlsym(handle, "CreatePlugin"));
    if (create_fn == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "dlsym(\"CreatePlugin\") failed for plugin '" + manifest.name + "'",
            std::source_location::current(),
        });
    }
    if (dlsym(handle, "DestroyPlugin") == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "dlsym(\"DestroyPlugin\") failed for plugin '" + manifest.name + "'",
            std::source_location::current(),
        });
    }

    IPlugin* raw_instance = create_fn();
    if (raw_instance == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "CreatePlugin() returned nullptr for plugin '" + manifest.name + "'",
            std::source_location::current(),
        });
    }
    return std::shared_ptr<IPlugin>(raw_instance, [](IPlugin*) noexcept {});
}

}  // namespace

// ── Inline validators (formerly VersionManager / LicenseManager / CapabilityManager) ──

auto PluginManager::CheckVersion(const VersionRange& required,
                                  const SemVer& actual) noexcept -> Result<void> {
    if (actual.major < required.min_inclusive.major ||
        (actual.major == required.min_inclusive.major &&
         actual.minor < required.min_inclusive.minor) ||
        actual.major > required.max_exclusive.major ||
        (actual.major == required.max_exclusive.major &&
         actual.minor >= required.max_exclusive.minor)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Plugin_VersionIncompatible,
            "plugin version incompatible with required range",
            std::source_location::current(),
        });
    }
    return {};
}

auto PluginManager::ValidateLicense(std::string_view token) noexcept -> Result<void> {
    if (token.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Plugin_LicenseInvalid,
            "plugin license token is empty",
            std::source_location::current(),
        });
    }
    return {};
}

auto PluginManager::RegisterKnownCapability(std::string capability) -> Result<void> {
    std::unique_lock lock(cap_mutex_);
    auto [it, inserted] = known_capabilities_.insert(std::move(capability));
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "capability already registered",
            std::source_location::current(),
        });
    }
    return {};
}

auto PluginManager::ValidateCapabilities(
    const std::vector<std::string>& declared) const -> Result<void> {
    std::shared_lock lock(cap_mutex_);
    for (const auto& cap : declared) {
        if (!known_capabilities_.contains(cap)) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Plugin_CapabilityUnsupported,
                "declared capability '" + cap + "' is not registered as known",
                std::source_location::current(),
            });
        }
    }
    return {};
}

// ── PluginManager ──────────────────────────────────────────────────────────

PluginManager::PluginManager(Context& context) noexcept : context_(context) {}

PluginManager::~PluginManager() {
    for (const auto& [name, handle] : library_handles_) {
        const auto type_id = detail::Fnv1aHash(name);
        auto resolved = plugin_registry_.Resolve(type_id);
        if (resolved.has_value()) {
            dlerror();
            auto* destroy_fn = reinterpret_cast<DestroyPluginFn>(
                dlsym(handle, "DestroyPlugin"));
            if (destroy_fn != nullptr) {
                destroy_fn(resolved->get());
            }
        }
        dlclose(handle);
    }
}

auto PluginManager::DiscoverManifests(const std::filesystem::path& plugin_dir) -> Result<void> {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(plugin_dir)) {
        if (entry.path().filename() != "plugin.yaml") continue;
        auto manifest_result = LoadManifest(entry.path());
        if (!manifest_result.has_value()) {
            return tl::make_unexpected(manifest_result.error());
        }
        PluginManifest manifest = std::move(manifest_result).value();
        manifest.library_path =
            (entry.path().parent_path() / manifest.library_path).string();
        manifest_index_[manifest.name] = std::move(manifest);
    }
    return {};
}

auto PluginManager::Load(const std::string& plugin_name) -> Result<void> {
    std::unordered_set<std::string> visiting;
    return EnsureLoaded(plugin_name, visiting);
}

auto PluginManager::Resolve(TypeId id) const -> Result<std::shared_ptr<IPlugin>> {
    return plugin_registry_.Resolve(id);
}

auto PluginManager::Shutdown() -> Result<void> {
    return ShutdownFrom(load_order_.size(), Result<void>{});
}

auto PluginManager::FindManifest(const std::string& plugin_name) const
    -> Result<PluginManifest> {
    auto it = manifest_index_.find(plugin_name);
    if (it == manifest_index_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeNotFound,
            "no manifest discovered for plugin '" + plugin_name + "'",
            std::source_location::current(),
        });
    }
    return it->second;
}

auto PluginManager::EnsureLoaded(const std::string& plugin_name,
                                  std::unordered_set<std::string>& visiting) -> Result<void> {
    const auto type_id = detail::Fnv1aHash(plugin_name);
    if (plugin_registry_.Resolve(type_id).has_value()) {
        return Result<void>{};
    }
    if (visiting.contains(plugin_name)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Plugin_CircularDependency,
            "circular dependency detected while resolving '" + plugin_name + "'",
            std::source_location::current(),
        });
    }
    visiting.insert(plugin_name);

    return FindManifest(plugin_name)
        .and_then([this, &visiting](const PluginManifest& manifest) -> Result<void> {
            return ResolveEach(manifest.dependencies, visiting)
                .and_then([this, &manifest] { return LoadSingle(manifest); });
        });
}

auto PluginManager::ResolveEach(const std::vector<PluginDependency>& deps,
                                 std::unordered_set<std::string>& visiting) -> Result<void> {
    if (deps.empty()) return Result<void>{};
    const auto& head = deps.front();
    return EnsureLoaded(head.plugin_name, visiting)
        .and_then([this, &head]() -> Result<void> {
            return plugin_registry_.Resolve(detail::Fnv1aHash(head.plugin_name))
                .and_then([this, &head](const std::shared_ptr<IPlugin>& instance) {
                    return CheckVersion(head.required_version,
                                         instance->GetManifest().version);
                });
        })
        .and_then([this, &deps, &visiting] {
            const std::vector<PluginDependency> rest(deps.begin() + 1, deps.end());
            return ResolveEach(rest, visiting);
        });
}

auto PluginManager::LoadSingle(const PluginManifest& manifest) -> Result<void> {
    return ValidateCapabilities(manifest.capabilities)
        .and_then([this, &manifest] { return ValidateLicense(manifest.license_token); })
        .and_then([this, &manifest]() -> Result<void*> { return DlOpen(manifest.library_path); })
        .and_then([this, &manifest](void* handle) -> Result<std::shared_ptr<IPlugin>> {
            library_handles_[manifest.name] = handle;
            return CreateInstance(handle, manifest);
        })
        .and_then([this, &manifest](std::shared_ptr<IPlugin> instance) -> Result<void> {
            const auto type_id = detail::Fnv1aHash(manifest.name);
            return plugin_registry_.Register(type_id, instance)
                .and_then([this, instance, type_id] {
                    return instance->OnInitialize(context_)
                        .and_then([instance, this, type_id] {
                            return instance->OnStart(context_)
                                .and_then([this, type_id] {
                                    load_order_.push_back(type_id);
                                    return Result<void>{};
                                });
                        });
                });
        });
}

auto PluginManager::ShutdownFrom(std::size_t remaining, Result<void> first_error)
    -> Result<void> {
    if (remaining == 0) return first_error;
    const auto type_id = load_order_[remaining - 1];
    auto next_error = plugin_registry_.Resolve(type_id)
        .and_then([this](const std::shared_ptr<IPlugin>& instance) {
            return instance->OnStop(context_);
        });
    return ShutdownFrom(remaining - 1,
                         first_error ? next_error : first_error);
}

}  // namespace sai
