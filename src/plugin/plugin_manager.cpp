#include <sai/plugin/plugin_manager.h>

#include <dlfcn.h>

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

// Constructs the plugin instance via the .so's exported CreatePlugin symbol.
// The returned shared_ptr's deleter is deliberately a no-op: destruction
// must go through the same .so's exported DestroyPlugin, and that call must
// happen strictly before the .so is dlclose()'d (see
// PluginManager::~PluginManager and 1.3-core-plugin-system.md §11 Memory).
// Registry<IPlugin> has no Unregister/Clear hook to force that ordering
// through the shared_ptr's own deleter, so ~PluginManager() drives
// DestroyPlugin explicitly and this deleter only needs to make the eventual,
// implicit destruction of plugin_registry_'s entries harmless.
auto CreateInstance(void* handle, const PluginManifest& manifest)
    -> Result<std::shared_ptr<IPlugin>> {
    dlerror();  // Clear any pending error before probing symbols.
    auto* create_fn = reinterpret_cast<CreatePluginFn>(dlsym(handle, "CreatePlugin"));
    if (create_fn == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "dlsym(\"CreatePlugin\") failed for plugin '" + manifest.name + "'",
            std::source_location::current(),
        });
    }
    // Fail fast if DestroyPlugin is missing too — a plugin we cannot ever
    // clean up should not be registered in the first place.
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

PluginManager::PluginManager(Context& context) noexcept : context_(context) {}

PluginManager::~PluginManager() {
    for (const auto& [name, handle] : library_handles_) {
        const auto type_id = detail::Fnv1aHash(name);
        auto resolved = plugin_registry_.Resolve(type_id);
        if (resolved.has_value()) {
            dlerror();
            auto* destroy_fn = reinterpret_cast<DestroyPluginFn>(dlsym(handle, "DestroyPlugin"));
            if (destroy_fn != nullptr) {
                destroy_fn(resolved->get());
            }
        }
        dlclose(handle);
    }
}

auto PluginManager::DiscoverManifests(const std::filesystem::path& plugin_dir) -> Result<void> {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(plugin_dir)) {
        if (entry.path().filename() != "plugin.yaml") {
            continue;
        }
        auto manifest_result = LoadManifest(entry.path());
        if (!manifest_result.has_value()) {
            return tl::make_unexpected(manifest_result.error());
        }
        PluginManifest manifest = std::move(manifest_result).value();
        // library_path in the manifest is relative to the manifest's own
        // directory (see manifest.h); resolve it to an absolute path here so
        // LoadSingle can dlopen it regardless of the process's current
        // working directory.
        manifest.library_path = (entry.path().parent_path() / manifest.library_path).string();
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

auto PluginManager::FindManifest(const std::string& plugin_name) const -> Result<PluginManifest> {
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
        return Result<void>{};  // Already loaded, idempotent success (a recursion base case).
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
    if (deps.empty()) {
        return Result<void>{};  // Recursion base case: no more dependencies.
    }
    const auto& head = deps.front();
    return EnsureLoaded(head.plugin_name, visiting)
        .and_then([this, &head]() -> Result<void> {
            return plugin_registry_.Resolve(detail::Fnv1aHash(head.plugin_name))
                .and_then([this, &head](const std::shared_ptr<IPlugin>& instance) {
                    return version_manager_.CheckCompatible(head.required_version,
                                                             instance->GetManifest().version);
                });
        })
        .and_then([this, &deps, &visiting] {
            const std::vector<PluginDependency> rest(deps.begin() + 1, deps.end());
            return ResolveEach(rest, visiting);
        });
}

auto PluginManager::LoadSingle(const PluginManifest& manifest) -> Result<void> {
    return capability_manager_.Validate(manifest.capabilities)
        .and_then([this, &manifest] { return license_manager_.Validate(manifest.license_token); })
        .and_then([this, &manifest]() -> Result<void*> { return DlOpen(manifest.library_path); })
        .and_then([this, &manifest](void* handle) -> Result<std::shared_ptr<IPlugin>> {
            library_handles_[manifest.name] = handle;
            return CreateInstance(handle, manifest);
        })
        .and_then([this, &manifest](std::shared_ptr<IPlugin> instance) -> Result<void> {
            const auto type_id = detail::Fnv1aHash(manifest.name);
            return plugin_registry_.Register(type_id, instance)
                .and_then([this, instance, type_id] {
                    // Never call Context::RegisterModule — Context must never
                    // hold a pointer to a plugin instance (see §3 Design).
                    // PluginManager drives the lifecycle hooks itself;
                    // `instance` is the same shared_ptr<IPlugin> just
                    // Register()'d into plugin_registry_, so converting it to
                    // IModule& below is a reference conversion only, no
                    // ownership transfer.
                    // load_order_ is appended only after OnStart succeeds. If
                    // OnInitialize or OnStart fails here, the instance stays in
                    // plugin_registry_ (still Resolve-able) but is NOT in
                    // load_order_, so Shutdown() will not call OnStop on it even
                    // though OnInitialize may already have run. This asymmetry is
                    // deliberate — consistent with 1.2's "no automatic rollback
                    // on assembly failure" decision; ~PluginManager still reclaims
                    // the instance via the exported DestroyPlugin path.
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

auto PluginManager::ShutdownFrom(std::size_t remaining, Result<void> first_error) -> Result<void> {
    if (remaining == 0) {
        return first_error;  // Recursion base case: every load_order_ entry processed.
    }
    const auto type_id = load_order_[remaining - 1];  // Tail to head: the reverse of load order.
    auto next_error = plugin_registry_.Resolve(type_id)
        .and_then([this](const std::shared_ptr<IPlugin>& instance) {
            return instance->OnStop(context_);
        });
    // Same error-handling policy as Context::Stop(): a single plugin's
    // OnStop failure does not block the remaining plugins, only the first
    // error encountered during the walk is kept.
    return ShutdownFrom(remaining - 1, first_error ? next_error : first_error);
}

}  // namespace sai
