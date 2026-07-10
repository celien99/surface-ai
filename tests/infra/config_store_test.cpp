#include <sai/infra/config_store.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include <sai/core/error.h>
#include <sai/infra/config_schema.h>

namespace {

using sai::ErrorCode;
using sai::Result;
using sai::infra::ConfigSchema;
using sai::infra::ConfigStore;

// A validator that always accepts, regardless of the node's content.
auto Accept() -> sai::infra::FieldValidator {
    return [](const YAML::Node&) -> Result<void> { return {}; };
}

// A validator that always rejects.
auto Reject() -> sai::infra::FieldValidator {
    return [](const YAML::Node&) -> Result<void> {
        return tl::make_unexpected(sai::ErrorInfo{
            ErrorCode::Infra_ConfigValidationFailed, "rejected",
            std::source_location::current()});
    };
}

// Requires the node to parse as a positive int.
auto ValidatePositiveInt() -> sai::infra::FieldValidator {
    return [](const YAML::Node& node) -> Result<void> {
        if (node.as<int>(0) > 0) {
            return {};
        }
        return tl::make_unexpected(sai::ErrorInfo{
            ErrorCode::Infra_ConfigValidationFailed, "not positive",
            std::source_location::current()});
    };
}

auto WriteTempFile(const std::string& name, const std::string& content)
    -> std::filesystem::path {
    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::trunc);
    out << content;
    out.close();
    return path;
}

constexpr const char* kValidYaml = R"(capture:
  camera_count: 4
  model: "acA1300"
memory:
  max_concurrent_frames: 8
)";

// ---------------------------------------------------------------------------
// ConfigSchema
// ---------------------------------------------------------------------------

TEST(ConfigSchemaTest, ValidateSucceedsWhenAllRulesSatisfied) {
    ConfigSchema schema;
    schema.RequireField("capture.camera_count", ValidatePositiveInt())
        .RequireField("memory.max_concurrent_frames", ValidatePositiveInt());

    const YAML::Node root = YAML::Load(kValidYaml);
    auto result = schema.Validate(root);
    EXPECT_TRUE(result.has_value()) << "all rules satisfied should pass";
}

TEST(ConfigSchemaTest, MissingRequiredFieldFailsValidation) {
    ConfigSchema schema;
    schema.RequireField("capture.does_not_exist", Accept());

    const YAML::Node root = YAML::Load(kValidYaml);
    auto result = schema.Validate(root);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigValidationFailed);
}

TEST(ConfigSchemaTest, FieldFailingValidatorClosureFailsValidation) {
    ConfigSchema schema;
    // Field exists but the closure rejects it.
    schema.RequireField("capture.camera_count", Reject());

    const YAML::Node root = YAML::Load(kValidYaml);
    auto result = schema.Validate(root);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigValidationFailed);
}

TEST(ConfigSchemaTest, ValidationStopsAtFirstFailure) {
    int second_invoked = 0;
    auto counting = [&second_invoked](const YAML::Node&) -> Result<void> {
        ++second_invoked;
        return {};
    };

    ConfigSchema schema;
    // First rule fails (field exists but rejected); second rule must never run.
    schema.RequireField("capture.camera_count", Reject())
        .RequireField("memory.max_concurrent_frames", counting);

    const YAML::Node root = YAML::Load(kValidYaml);
    auto result = schema.Validate(root);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigValidationFailed);
    EXPECT_EQ(second_invoked, 0)
        << "validator after the first failure must not be invoked";
}

// ---------------------------------------------------------------------------
// ConfigStore::Load / Get
// ---------------------------------------------------------------------------

TEST(ConfigStoreTest, LoadValidYamlWithAcceptingSchemaSucceeds) {
    ConfigSchema schema;
    schema.RequireField("capture.camera_count", ValidatePositiveInt());
    ConfigStore store(std::move(schema));

    auto path = WriteTempFile("sai_cfg_valid.yaml", kValidYaml);
    auto result = store.Load(path);
    EXPECT_TRUE(result.has_value()) << "valid + accepted config should load";
}

TEST(ConfigStoreTest, LoadNonexistentPathReturnsFileNotFound) {
    ConfigStore store(ConfigSchema{});
    auto result = store.Load("/no/such/sai_config_file_zzz.yaml");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigFileNotFound);
}

TEST(ConfigStoreTest, LoadInvalidYamlReturnsParseError) {
    ConfigStore store(ConfigSchema{});
    // Unbalanced flow bracket -> yaml-cpp parse failure.
    auto path = WriteTempFile("sai_cfg_bad.yaml", "capture: [1, 2, 3\nfoo: :bar:\n");
    auto result = store.Load(path);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigParseError);
}

TEST(ConfigStoreTest, LoadValidYamlFailingSchemaReturnsValidationFailedAndLeavesStoreUnchanged) {
    ConfigSchema schema;
    schema.RequireField("capture.camera_count", Reject());
    ConfigStore store(std::move(schema));

    auto path = WriteTempFile("sai_cfg_reject.yaml", kValidYaml);
    auto result = store.Load(path);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Infra_ConfigValidationFailed);

    // First load failed: internal tree stayed empty. A subsequent Get must fail
    // cleanly (key-not-found), not crash.
    auto value = store.Get<int>("capture.camera_count");
    ASSERT_FALSE(value.has_value());
    EXPECT_EQ(value.error().code, ErrorCode::Infra_ConfigKeyNotFound);
}

TEST(ConfigStoreTest, GetReturnsValueAndTypeForExistingKey) {
    ConfigSchema schema;
    ConfigStore store(std::move(schema));
    auto path = WriteTempFile("sai_cfg_get.yaml", kValidYaml);
    ASSERT_TRUE(store.Load(path).has_value());

    auto count = store.Get<int>("capture.camera_count");
    ASSERT_TRUE(count.has_value());
    EXPECT_EQ(count.value(), 4);

    auto model = store.Get<std::string>("capture.model");
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(model.value(), "acA1300");

    auto frames = store.Get<int>("memory.max_concurrent_frames");
    ASSERT_TRUE(frames.has_value());
    EXPECT_EQ(frames.value(), 8);
}

TEST(ConfigStoreTest, GetMissingKeyReturnsKeyNotFound) {
    ConfigStore store(ConfigSchema{});
    auto path = WriteTempFile("sai_cfg_missing.yaml", kValidYaml);
    ASSERT_TRUE(store.Load(path).has_value());

    auto missing = store.Get<int>("capture.nonexistent");
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, ErrorCode::Infra_ConfigKeyNotFound);

    // A path that dives through a scalar is also key-not-found, not a crash.
    auto deep = store.Get<int>("capture.camera_count.deeper");
    ASSERT_FALSE(deep.has_value());
    EXPECT_EQ(deep.error().code, ErrorCode::Infra_ConfigKeyNotFound);
}

TEST(ConfigStoreTest, GetIncompatibleTypeReturnsTypeMismatch) {
    ConfigStore store(ConfigSchema{});
    auto path = WriteTempFile("sai_cfg_mismatch.yaml", kValidYaml);
    ASSERT_TRUE(store.Load(path).has_value());

    // "acA1300" is not convertible to int.
    auto bad = store.Get<int>("capture.model");
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error().code, ErrorCode::Infra_ConfigKeyTypeMismatch);
}

}  // namespace
