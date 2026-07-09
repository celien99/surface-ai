#include <sai/plugin/capability_manager.h>
#include <sai/plugin/license_manager.h>
#include <sai/plugin/manifest.h>
#include <sai/plugin/version_manager.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace {

constexpr sai::SemVer kMinInclusive{.major = 1, .minor = 2, .patch = 0};
constexpr sai::SemVer kMaxExclusive{.major = 2, .minor = 0, .patch = 0};
constexpr sai::VersionRange kRange{.min_inclusive = kMinInclusive, .max_exclusive = kMaxExclusive};

}  // namespace

// --- VersionManager -------------------------------------------------------

TEST(VersionManagerTest, InRangeVersionIsAccepted) {
    auto result = sai::VersionManager::CheckCompatible(kRange, sai::SemVer{1, 5, 3});

    EXPECT_TRUE(result.has_value());
}

TEST(VersionManagerTest, BelowMinInclusiveIsRejected) {
    auto result = sai::VersionManager::CheckCompatible(kRange, sai::SemVer{1, 1, 9});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Plugin_VersionIncompatible);
}

TEST(VersionManagerTest, ExactlyMinInclusiveIsAccepted) {
    auto result = sai::VersionManager::CheckCompatible(kRange, kMinInclusive);

    EXPECT_TRUE(result.has_value());
}

TEST(VersionManagerTest, ExactlyMaxExclusiveIsRejected) {
    auto result = sai::VersionManager::CheckCompatible(kRange, kMaxExclusive);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Plugin_VersionIncompatible);
}

TEST(VersionManagerTest, AboveMaxExclusiveIsRejected) {
    auto result = sai::VersionManager::CheckCompatible(kRange, sai::SemVer{2, 3, 0});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Plugin_VersionIncompatible);
}

// --- CapabilityManager ------------------------------------------------------

TEST(CapabilityManagerTest, ValidateSubsetOfRegisteredCapabilitiesSucceeds) {
    sai::CapabilityManager manager;
    ASSERT_TRUE(manager.RegisterKnownCapability("camera.gige").has_value());
    ASSERT_TRUE(manager.RegisterKnownCapability("detector.onnx").has_value());

    auto result = manager.Validate({"camera.gige"});

    EXPECT_TRUE(result.has_value());
}

TEST(CapabilityManagerTest, ValidateUnregisteredCapabilityReturnsCapabilityUnsupported) {
    sai::CapabilityManager manager;
    ASSERT_TRUE(manager.RegisterKnownCapability("camera.gige").has_value());

    auto result = manager.Validate({"detector.onnx"});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Plugin_CapabilityUnsupported);
}

TEST(CapabilityManagerTest, DuplicateRegisterReturnsTypeAlreadyRegistered) {
    sai::CapabilityManager manager;
    ASSERT_TRUE(manager.RegisterKnownCapability("camera.gige").has_value());

    auto second = manager.RegisterKnownCapability("camera.gige");

    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, sai::ErrorCode::Core_TypeAlreadyRegistered);
}

// --- LicenseManager ----------------------------------------------------------

TEST(LicenseManagerTest, NonEmptyTokenIsValid) {
    sai::LicenseManager manager;

    auto result = manager.Validate("some-license-token");

    EXPECT_TRUE(result.has_value());
}

TEST(LicenseManagerTest, EmptyTokenIsInvalid) {
    sai::LicenseManager manager;

    auto result = manager.Validate("");

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Plugin_LicenseInvalid);
}

// --- PluginManifest YAML deserialization -------------------------------------

namespace {

class PluginManifestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = std::filesystem::temp_directory_path() /
                "sai_plugin_manifest_validators_test_plugin.yaml";
        std::ofstream out(path_);
        out << R"(name: sai.detector.onnx
library_path: lib/libdetector.so
version:
  major: 1
  minor: 4
  patch: 2
capabilities:
  - detector.onnx
  - detector.gpu
dependencies:
  - plugin_name: sai.inference-runtime
    required_version:
      min_inclusive:
        major: 1
        minor: 0
        patch: 0
      max_exclusive:
        major: 2
        minor: 0
        patch: 0
license_token: deployed-license-abc123
)";
    }

    void TearDown() override { std::filesystem::remove(path_); }

    std::filesystem::path path_;
};

}  // namespace

TEST_F(PluginManifestFixture, LoadManifestRoundTripsAllFields) {
    auto result = sai::LoadManifest(path_);

    ASSERT_TRUE(result.has_value());
    const sai::PluginManifest& manifest = *result;

    EXPECT_EQ(manifest.name, "sai.detector.onnx");
    EXPECT_EQ(manifest.library_path, "lib/libdetector.so");
    EXPECT_EQ(manifest.version.major, 1u);
    EXPECT_EQ(manifest.version.minor, 4u);
    EXPECT_EQ(manifest.version.patch, 2u);
    EXPECT_EQ(manifest.license_token, "deployed-license-abc123");

    ASSERT_EQ(manifest.capabilities.size(), 2u);
    EXPECT_EQ(manifest.capabilities[0], "detector.onnx");
    EXPECT_EQ(manifest.capabilities[1], "detector.gpu");

    ASSERT_EQ(manifest.dependencies.size(), 1u);
    const sai::PluginDependency& dependency = manifest.dependencies[0];
    EXPECT_EQ(dependency.plugin_name, "sai.inference-runtime");
    EXPECT_EQ(dependency.required_version.min_inclusive.major, 1u);
    EXPECT_EQ(dependency.required_version.min_inclusive.minor, 0u);
    EXPECT_EQ(dependency.required_version.min_inclusive.patch, 0u);
    EXPECT_EQ(dependency.required_version.max_exclusive.major, 2u);
    EXPECT_EQ(dependency.required_version.max_exclusive.minor, 0u);
    EXPECT_EQ(dependency.required_version.max_exclusive.patch, 0u);
}

TEST(LoadManifestTest, MissingFileReturnsError) {
    auto result = sai::LoadManifest(std::filesystem::path("/nonexistent/plugin.yaml"));

    EXPECT_FALSE(result.has_value());
}
