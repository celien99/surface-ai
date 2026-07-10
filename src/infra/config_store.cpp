#include <sai/infra/config_store.h>

#include <mutex>
#include <source_location>
#include <string>
#include <utility>

namespace sai::infra {

ConfigStore::ConfigStore(ConfigSchema schema) noexcept
    : schema_(std::move(schema)) {}

ConfigStore::~ConfigStore() noexcept = default;

auto ConfigStore::Load(std::filesystem::path path) -> Result<void> {
    YAML::Node parsed;
    try {
        parsed = YAML::LoadFile(path.string());
    } catch (const YAML::BadFile& ex) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigFileNotFound,
            "config file not found: " + path.string(),
            std::source_location::current(),
        });
    } catch (const YAML::ParserException& ex) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigParseError,
            "config parse error in " + path.string() + ": " + ex.what(),
            std::source_location::current(),
        });
    }

    // Validate against the schema before touching internal state; on failure the
    // held config tree stays exactly as it was (empty on a first failed load).
    auto validated = schema_.Validate(parsed);
    if (!validated) {
        return validated;
    }

    std::unique_lock lock(mutex_);
    root_ = parsed;
    path_ = std::move(path);
    return {};
}

// NOTE: EnableHotReload is declared in config_store.h but deliberately NOT
// defined here — it is the inotify-based, Linux-only path defined by Task 10's
// config_store_inotify.cpp. On this portable build it is declared-but-undefined,
// which is fine because nothing in this subset (or its tests) ODR-uses it.

}  // namespace sai::infra
