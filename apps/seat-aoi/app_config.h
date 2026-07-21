#pragma once

#include <cstddef>
#include <filesystem>

namespace seat_aoi::config {

// Walk up from the executable to find the project root.
// Uses a sentinel file (apps/seat-aoi/resources/pipeline.yaml) to detect the root.
// Falls back to walking up 4 levels (build/<preset>/apps/seat-aoi) if sentinel not found.
inline auto ProjectRoot() -> std::filesystem::path {
    static const auto root = [] {
        std::error_code ec;
        auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        auto dir = ec ? std::filesystem::current_path() : exe.parent_path();

        // Search upward for the sentinel: apps/seat-aoi/resources/pipeline.yaml
        for (auto d = dir; d.has_parent_path(); d = d.parent_path()) {
            if (std::filesystem::exists(d / "apps/seat-aoi/resources/pipeline.yaml")) {
                return d;
            }
        }

        // Fallback: binary is at build/<preset>/apps/seat-aoi/seat_aoi
        auto fallback = dir;
        for (int i = 0; i < 4 && fallback.has_parent_path(); ++i)
            fallback = fallback.parent_path();
        return fallback;
    }();
    return root;
}

// ── Resource paths (resolved relative to project root) ──
inline auto PipelineYaml() -> std::filesystem::path { return ProjectRoot() / "apps/seat-aoi/resources/pipeline.yaml"; }
inline auto DinoV2Engine() -> std::filesystem::path { return ProjectRoot() / "resources/models/dino_v2_vit_base.engine"; }
inline auto ClipEngine()   -> std::filesystem::path { return ProjectRoot() / "resources/models/clip_vit_b32.engine"; }
inline auto Sam2Engine()   -> std::filesystem::path { return ProjectRoot() / "resources/models/sam2_vit_h.engine"; }
inline auto DecisionTree() -> std::filesystem::path { return ProjectRoot() / "apps/seat-aoi/resources/trees/seat_leather_inspection.yaml"; }
inline auto TuningYaml()   -> std::filesystem::path { return ProjectRoot() / "apps/seat-aoi/resources/tuning/seat_leather_tuning.yaml"; }

// ── Model / preprocessing constants ──
constexpr std::size_t kEmbedDim  = 768;
constexpr std::size_t kImageSize = 518;
constexpr std::size_t kPatchSize = 14;

// ── Defaults ──
constexpr const char* kDefaultOutputDir = "/tmp/surface-ai/results/";

}  // namespace seat_aoi::config
