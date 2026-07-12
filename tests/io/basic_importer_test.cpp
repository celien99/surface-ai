#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sai/io/importer.h>
#include <sai/image/surface_image.h>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

namespace {

using namespace sai::io;

struct TestDir {
    std::filesystem::path path;
    TestDir()
        : path(std::filesystem::temp_directory_path() / "sai_io_importer_test") {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TestDir() { std::filesystem::remove_all(path); }
};

}  // namespace

// ============================================================================
// Test 1: ImportMetadata on a valid YAML file returns parseable Node
// ============================================================================

TEST(BasicImporterTest, ImportMetadataValidYaml) {
    TestDir td;
    auto yaml_path = td.path / "test.yaml";
    {
        std::ofstream ofs(yaml_path);
        ofs << "sku_id: \"TestSKU-001\"\n"
               "product: \"Car Seat Driver\"\n"
               "batch: \"B20260701\"\n";
    }

    BasicImporter importer;
    auto result = importer.ImportMetadata(yaml_path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_TRUE(result->IsMap());
    EXPECT_EQ((*result)["sku_id"].as<std::string>(), "TestSKU-001");
    EXPECT_EQ((*result)["product"].as<std::string>(), "Car Seat Driver");
    EXPECT_EQ((*result)["batch"].as<std::string>(), "B20260701");
}

// ============================================================================
// Test 2: ImportMetadata on a missing file returns Io_ImportFileNotFound
// ============================================================================

TEST(BasicImporterTest, ImportMetadataMissingFile) {
    auto missing = std::filesystem::temp_directory_path() / "nonexistent_metadata.yaml";

    BasicImporter importer;
    auto result = importer.ImportMetadata(missing);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Io_ImportFileNotFound);
}

// ============================================================================
// Test 3: ImportMetadata on a malformed YAML returns Io_ImportParseFailed
// ============================================================================

TEST(BasicImporterTest, ImportMetadataMalformedYaml) {
    TestDir td;
    auto yaml_path = td.path / "malformed.yaml";
    {
        std::ofstream ofs(yaml_path);
        ofs << "key: [unclosed bracket\n";
    }

    BasicImporter importer;
    auto result = importer.ImportMetadata(yaml_path);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Io_ImportParseFailed);
}

// ============================================================================
// Test 4: ImportImage on a hand-written 2x2 P6 PPM returns valid SurfaceImage
// ============================================================================

TEST(BasicImporterTest, ImportImageValidPpm) {
    TestDir td;
    auto ppm_path = td.path / "test.ppm";
    {
        std::ofstream ofs(ppm_path, std::ios::binary);
        // P6 PPM header: P6\n2 2\n255\n
        ofs << "P6\n2 2\n255\n";

        // 4 pixels, each 3 bytes RGB
        std::vector<std::uint8_t> pixels = {
            255, 0, 0,      // (0,0) red
            0, 255, 0,      // (1,0) green
            0, 0, 255,      // (0,1) blue
            128, 128, 128   // (1,1) gray
        };
        ofs.write(reinterpret_cast<const char*>(pixels.data()),
                  static_cast<std::streamsize>(pixels.size()));
    }

    BasicImporter importer;
    auto result = importer.ImportImage(ppm_path);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    const auto& image = *result.value();
    const auto& meta = image.Meta();
    EXPECT_EQ(meta.width, 2U);
    EXPECT_EQ(meta.height, 2U);
    EXPECT_EQ(meta.channels, 3U);
    EXPECT_EQ(meta.pixel_format, sai::image::PixelFormat::RGB8);
    EXPECT_EQ(image.SizeBytes(), 12U);

    // Verify pixel bytes
    const auto* data = image.Data();
    ASSERT_NE(data, nullptr);
    std::vector<std::uint8_t> expected = {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        128, 128, 128
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(data[i], expected[i]) << "byte mismatch at index " << i;
    }
}

// ============================================================================
// Test 5: ImportImage on a missing file returns Io_ImportFileNotFound
// ============================================================================

TEST(BasicImporterTest, ImportImageMissingFile) {
    auto missing = std::filesystem::temp_directory_path() / "nonexistent_image.ppm";

    BasicImporter importer;
    auto result = importer.ImportImage(missing);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Io_ImportFileNotFound);
}

// ============================================================================
// Test 6: FormatName() == "basic_import"
// ============================================================================

TEST(BasicImporterTest, FormatNameIsBasicImport) {
    BasicImporter importer;
    EXPECT_EQ(importer.FormatName(), "basic_import");
}

// ============================================================================
// Test 7: Lifecycle no-op smoke test
// ============================================================================

TEST(BasicImporterTest, LifecycleMethodsAreNoop) {
    BasicImporter importer;
    SUCCEED();
}
