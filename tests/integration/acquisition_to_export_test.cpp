#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <sai/core/error.h>
#include <sai/device/ring_buffer.h>
#include <sai/image/image.h>
#include <sai/image/preprocess.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/io/exporter.h>

#include "fake_camera.h"

namespace {

using sai::image::Compose;
using sai::image::ImageMeta;
using sai::image::MakeDebayer;
using sai::image::MakeFlatField;
using sai::image::MakeResize;
using sai::image::MakeWhiteBalance;
using sai::image::PixelFormat;
using sai::image::RawImage;
using sai::image::SurfaceImage;
using sai::io::DefectRecord;
using sai::io::InspectionResult;
using sai::io::JsonExporter;
using sai::test::FakeCamera;

// Helper: build a BayerRG8 ImageMeta.
auto BayerRG8Meta(std::size_t width, std::size_t height) -> ImageMeta {
    ImageMeta m;
    m.width = width;
    m.height = height;
    m.channels = 1;
    m.pixel_format = PixelFormat::BayerRG8;
    return m;
}

}  // namespace

// Milestone-2 acceptance gate: camera frame → RingBuffer → preprocess chain →
// SurfaceImage → InspectionResult → JsonExporter → result.json on disk.
TEST(AcquisitionToExport, SoftwareTriggerToJsonReport) {
    // 1. Setup camera
    FakeCamera cam;
    ASSERT_TRUE(cam.Connect().has_value());

    // 2. Ring buffer (capacity 4)
    sai::device::RingBuffer<RawImage> ring(4);

    // 3. Register callback: push every frame into the ring buffer
    ASSERT_TRUE(cam.RegisterFrameCallback([&](RawImage f) {
        ring.Push(std::move(f));
    }).has_value());

    // 4. Start software-triggered acquisition
    ASSERT_TRUE(cam.SetTriggerMode(sai::device::ICamera::TriggerMode::Software).has_value());
    ASSERT_TRUE(cam.StartAcquisition().has_value());

    // 5. Fire one software trigger → one frame pushed to ring buffer
    ASSERT_TRUE(cam.TriggerSoftware().has_value());

    // 6. Pop the frame
    auto raw_opt = ring.TryPop();
    ASSERT_TRUE(raw_opt.has_value());
    auto raw = std::make_unique<RawImage>(std::move(*raw_opt));
    ASSERT_EQ(raw->Meta().pixel_format, PixelFormat::BayerRG8);
    ASSERT_EQ(raw->Meta().width, 8u);
    ASSERT_EQ(raw->Meta().height, 8u);

    // 7. Build correction frame (BayerRG8 8x8, uniform 128 → identity gain)
    RawImage corr = RawImage::FromOwnedBuffer(
        std::vector<std::uint8_t>(64, 128), BayerRG8Meta(8, 8));

    // 8. Build and run preprocess chain:
    //    FlatField (identity) → Debayer (BayerRG8→RGB8) → WhiteBalance (1,1,1 pass-through) → Resize (8x8→4x4)
    auto chain = Compose({
        MakeFlatField(corr),
        MakeDebayer(),
        MakeWhiteBalance(1.0f, 1.0f, 1.0f),
        MakeResize(4, 4),
    });

    auto surf_result = chain(std::move(raw));
    ASSERT_TRUE(surf_result.has_value());

    auto surf = std::move(*surf_result);
    ASSERT_NE(surf, nullptr);
    EXPECT_EQ(surf->Meta().pixel_format, PixelFormat::RGB8);
    EXPECT_EQ(surf->Meta().width, 4u);
    EXPECT_EQ(surf->Meta().height, 4u);
    EXPECT_EQ(surf->Meta().channels, 3u);

    // 9. Build InspectionResult
    InspectionResult result;
    result.sku_id = "Tesla_Model3_Driver";
    result.serial_number = "SN-0001";
    result.timestamp = std::chrono::system_clock::now();
    result.verdict = "PASS";

    DefectRecord defect;
    defect.label = "scratch";
    defect.severity = "MINOR";
    defect.confidence = 0.95F;
    defect.location = {10, 20, 30, 40};
    defect.evidence_path = "/tmp/defect_001.png";
    result.defects.push_back(std::move(defect));

    // 10. Export to temp directory
    auto tmp_dir =
        std::filesystem::temp_directory_path() / "sai_integration_test";
    std::filesystem::remove_all(tmp_dir);  // clean start

    JsonExporter exporter;
    auto export_result = exporter.Export(result, tmp_dir,
                                         static_cast<const SurfaceImage*>(surf.get()));
    ASSERT_TRUE(export_result.has_value());

    // 11. Verify result.json exists and round-trips
    auto json_path = tmp_dir / "Tesla_Model3_Driver" / "SN-0001" / "result.json";
    ASSERT_TRUE(std::filesystem::exists(json_path));

    std::ifstream ifs(json_path);
    ASSERT_TRUE(ifs.is_open());
    nlohmann::json j = nlohmann::json::parse(ifs);

    EXPECT_EQ(j["sku_id"], "Tesla_Model3_Driver");
    EXPECT_EQ(j["serial_number"], "SN-0001");
    EXPECT_EQ(j["verdict"], "PASS");
    EXPECT_EQ(j["defects"].size(), 1u);
    EXPECT_EQ(j["defects"][0]["label"], "scratch");
    EXPECT_EQ(j["defects"][0]["severity"], "MINOR");
    EXPECT_EQ(j["defects"][0]["confidence"], 0.95F);
    EXPECT_EQ(j["defects"][0]["location"]["x"], 10);
    EXPECT_EQ(j["defects"][0]["location"]["y"], 20);
    EXPECT_EQ(j["defects"][0]["location"]["width"], 30);
    EXPECT_EQ(j["defects"][0]["location"]["height"], 40);

    // 12. Verify annotated.ppm was also written
    auto ppm_path = tmp_dir / "Tesla_Model3_Driver" / "SN-0001" / "annotated.ppm";
    EXPECT_TRUE(std::filesystem::exists(ppm_path));

    // Cleanup
    std::filesystem::remove_all(tmp_dir);
}
