#pragma once

#include <cstddef>

namespace seat_aoi {
namespace config {

// ── Resource paths ──
constexpr const char* kPipelineYaml = "resources/pipeline.yaml";
constexpr const char* kDinoV3Engine = "resources/models/dino_v3_vit_base.engine";
constexpr const char* kClipEngine   = "resources/models/clip_vit_b32.engine";
constexpr const char* kSam2Engine   = "resources/models/sam2_vit_h.engine";
constexpr const char* kDecisionTree = "resources/trees/seat_leather_inspection.yaml";
constexpr const char* kTuningYaml   = "resources/tuning/seat_leather_tuning.yaml";

// ── Defaults ──
constexpr const char* kDefaultCoresetOutput = "resources/coreset.bin";
constexpr const char* kDefaultOutputDir     = "/tmp/surface-ai/results/";

// ── Model / preprocessing constants ──
constexpr std::size_t kEmbedDim  = 1024;
constexpr std::size_t kImageSize = 1024;
constexpr std::size_t kPatchSize = 14;

}  // namespace config
}  // namespace seat_aoi
