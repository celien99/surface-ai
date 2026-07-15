#include <sai/io/coreset_manifest.h>

#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::io {

auto LoadCoresetManifest(const std::filesystem::path& yaml_path) noexcept
    -> Result<CoresetManifest> {
    // 1. Existence check
    std::error_code ec;
    if (!std::filesystem::exists(yaml_path, ec) || ec) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Coreset manifest not found: " + yaml_path.string(),
        });
    }

    // 2. YAML parse
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path.string());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = std::string("Failed to parse coreset manifest YAML: ") + e.what(),
        });
    }

    // 3. Validate top-level keys
    if (!root["surface"].IsDefined() || !root["banks"].IsDefined()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Coreset manifest must contain 'surface' and 'banks' keys",
        });
    }

    // 4. Extract surface_id
    std::string surface_id;
    try {
        surface_id = root["surface"].as<std::string>();
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = std::string("Failed to parse 'surface' as string: ") + e.what(),
        });
    }

    // 5. Parse banks sequence
    auto banks_node = root["banks"];
    if (!banks_node.IsSequence()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "'banks' must be a sequence",
        });
    }

    auto base_dir = yaml_path.parent_path();
    std::vector<CoresetBankEntry> banks;
    banks.reserve(banks_node.size());

    for (auto bank_node : banks_node) {
        CoresetBankEntry entry;

        if (!bank_node["position"].IsDefined() || !bank_node["path"].IsDefined()) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Io_ImportParseFailed,
                .message = "Each bank entry must have 'position' and 'path' fields",
            });
        }

        try {
            entry.position_id = bank_node["position"].as<std::uint16_t>();
        } catch (const YAML::Exception& e) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Io_ImportParseFailed,
                .message = std::string("Failed to parse bank 'position' as uint16: ") + e.what(),
            });
        }

        try {
            entry.path = base_dir / bank_node["path"].as<std::string>();
        } catch (const YAML::Exception& e) {
            return tl::make_unexpected(ErrorInfo{
                .code = ErrorCode::Io_ImportParseFailed,
                .message = std::string("Failed to parse bank 'path' as string: ") + e.what(),
            });
        }

        banks.push_back(std::move(entry));
    }

    return CoresetManifest{
        .surface_id = std::move(surface_id),
        .banks = std::move(banks),
    };
}

auto SaveCoresetManifest(const std::filesystem::path& yaml_path,
                          const CoresetManifest& manifest) noexcept
    -> Result<void> {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "surface" << YAML::Value << manifest.surface_id;

    out << YAML::Key << "banks" << YAML::Value << YAML::BeginSeq;
    for (const auto& bank : manifest.banks) {
        out << YAML::BeginMap;
        out << YAML::Key << "position" << YAML::Value << bank.position_id;
        out << YAML::Key << "path" << YAML::Value << bank.path.filename().string();
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;

    // Write to file
    std::ofstream out_file(yaml_path);
    if (!out_file) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ExportPathCreateFailed,
            .message = "Cannot open coreset manifest for writing: " + yaml_path.string(),
        });
    }
    out_file << out.c_str();
    out_file.close();

    if (!out_file) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_SerializationFailed,
            .message = "Failed to write coreset manifest: " + yaml_path.string(),
        });
    }

    return {};
}

}  // namespace sai::io
