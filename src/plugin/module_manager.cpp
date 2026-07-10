#include <sai/plugin/module_manager.h>

namespace sai {

ModuleManager::ModuleManager(Context& context) noexcept : context_(context) {}

auto ModuleManager::RegisterBuiltin(std::unique_ptr<IModule> module) -> Result<void> {
    return context_.RegisterModule(std::move(module));
}

}  // namespace sai
