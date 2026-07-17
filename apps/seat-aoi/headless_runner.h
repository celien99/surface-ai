#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct CliArgs;
struct AssembledApp;

// Per-frame record accumulated during headless processing.
// Used for both metrics computation and review_index.json generation.
struct FrameRecord {
    int frame_id;
    std::string image_path;        // original image path from dataset entry
    std::string verdict;           // "OK" | "NG" | "WARN" | "UNCERTAIN"
    double severity;
    double confidence;
    std::string expected_verdict;  // "" if not available
    std::uint16_t position_id;
    std::uint16_t light_id;
    std::string surface_id;
};

int RunHeadless(const CliArgs& cli, AssembledApp& app);

// Write review_index.json for GUI review mode consumption.
void WriteReviewIndex(std::string_view output_dir,
                      const std::vector<FrameRecord>& records,
                      std::string_view surface_id);
