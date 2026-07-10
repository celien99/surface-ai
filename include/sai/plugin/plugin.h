#pragma once

#include <sai/core/module.h>
#include <sai/core/reflectable.h>
#include <sai/plugin/manifest.h>

namespace sai {

// IPlugin derives IModule (reuses the 1.2 lifecycle hooks, no second state
// machine) and IReflectable (reuses the 1.1 TypeId mechanism as the key type
// for Registry<IPlugin>, see 1.3-core-plugin-system.md §3/§4). Beyond the
// lifecycle hooks it only adds one read-only accessor: exposes the manifest
// that has already passed the three validation steps at load time, for the
// rest of the framework (e.g. a diagnostics endpoint listing loaded plugin
// versions) to query — it never re-triggers any validation.
class IPlugin : public IModule, public IReflectable {
public:
    virtual ~IPlugin() = default;

    [[nodiscard]] virtual auto GetManifest() const noexcept -> const PluginManifest& = 0;
};

// Every plugin .so must export this pair of C symbols (extern "C" avoids
// C++ name mangling differing across compilers); PluginManager locates them
// via dlsym to construct and destroy the instance. Destruction must go
// through the same .so's exported DestroyPlugin — callers must never delete
// across a .so boundary themselves, see §11 Memory.
extern "C" using CreatePluginFn = IPlugin* (*)();
extern "C" using DestroyPluginFn = void (*)(IPlugin*);

}  // namespace sai
