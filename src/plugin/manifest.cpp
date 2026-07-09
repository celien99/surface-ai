#include <sai/plugin/manifest.h>

#include <source_location>

#include <yaml-cpp/yaml.h>

namespace sai {

namespace {

auto ParseSemVer(const YAML::Node& node) -> SemVer {
    return SemVer{
        .major = node["major"].as<std::uint32_t>(),
        .minor = node["minor"].as<std::uint32_t>(),
        .patch = node["patch"].as<std::uint32_t>(),
    };
}

auto ParseDependency(const YAML::Node& node) -> PluginDependency {
    const auto& range_node = node["required_version"];
    return PluginDependency{
        .plugin_name = node["plugin_name"].as<std::string>(),
        .required_version = VersionRange{
            .min_inclusive = ParseSemVer(range_node["min_inclusive"]),
            .max_exclusive = ParseSemVer(range_node["max_exclusive"]),
        },
    };
}

auto ParseManifest(const YAML::Node& root) -> PluginManifest {
    PluginManifest manifest;
    manifest.name = root["name"].as<std::string>();
    manifest.library_path = root["library_path"].as<std::string>();
    manifest.version = ParseSemVer(root["version"]);
    manifest.license_token = root["license_token"].as<std::string>();

    for (const auto& capability_node : root["capabilities"]) {
        manifest.capabilities.push_back(capability_node.as<std::string>());
    }
    for (const auto& dependency_node : root["dependencies"]) {
        manifest.dependencies.push_back(ParseDependency(dependency_node));
    }
    return manifest;
}

}  // namespace

auto LoadManifest(const std::filesystem::path& manifest_path) -> Result<PluginManifest> {
    try {
        const YAML::Node root = YAML::LoadFile(manifest_path.string());
        return ParseManifest(root);
    } catch (const YAML::Exception& ex) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            std::string("failed to parse plugin manifest: ") + ex.what(),
            std::source_location::current(),
        });
    }
}

}  // namespace sai
