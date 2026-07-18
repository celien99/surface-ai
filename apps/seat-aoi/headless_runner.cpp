#include "headless_runner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <sai/core/context.h>
#include <sai/detection/coreset_evolution.h>
#include <sai/pipeline/pipeline.h>
#include <sai/tuning/tuning_scheduler.h>
#include "app_builder.h"
#include "cli_args.h"

#include <nlohmann/json.hpp>

#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/reasoner/reasoner.h>

namespace {

struct MetricsCounts {
    int tp = 0;  // actual OK -> predicted OK
    int fp = 0;  // actual NG -> predicted OK
    int tn = 0;  // actual NG -> predicted NG
    int fn = 0;  // actual OK -> predicted NG
    // WARN and UNCERTAIN are tracked separately (neither TP nor FP)
    int actual_ok = 0;
    int actual_ng = 0;

    void Add(std::string_view predicted, std::string_view expected) {
        if (expected == "OK") {
            ++actual_ok;
            if (predicted == "OK") ++tp;
            else if (predicted == "NG") ++fn;
            // WARN / UNCERTAIN on OK -> not counted in binary matrix
        } else if (expected == "NG") {
            ++actual_ng;
            if (predicted == "NG") ++tn;
            else if (predicted == "OK") ++fp;
            // WARN / UNCERTAIN on NG -> not counted in binary matrix
        }
    }

    [[nodiscard]] auto Precision() const -> double {
        int denom = tp + fp;
        return denom > 0 ? static_cast<double>(tp) / static_cast<double>(denom) : 0.0;
    }
    [[nodiscard]] auto Recall() const -> double {
        int denom = tp + fn;
        return denom > 0 ? static_cast<double>(tp) / static_cast<double>(denom) : 0.0;
    }
    [[nodiscard]] auto F1() const -> double {
        double p = Precision(), r = Recall();
        return (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;
    }
    [[nodiscard]] auto Accuracy() const -> double {
        int denom = tp + tn + fp + fn;
        return denom > 0 ? static_cast<double>(tp + tn) / static_cast<double>(denom) : 0.0;
    }
};

}  // namespace

auto RunHeadless(const CliArgs& cli, AssembledApp& app) -> int {
    using namespace sai;

    std::vector<FrameRecord> records;

    auto process_entries = [&](auto& entries, io::BasicImporter& importer) -> int {
        int ok = 0, ng = 0, warn = 0, uncertain = 0, failed = 0;

        // Per-surface aggregation: track worst verdict for each product
        std::map<std::string, std::string> surface_verdict;
        std::map<std::string, int> surface_frame_count;

        for (size_t i = 0; i < entries.size(); ++i) {
            auto& entry = entries[i];
            auto img_result = importer.ImportImage(entry.path);
            if (!img_result) {
                std::cerr << "FAIL import: " << entry.path.filename() << "\n";
                ++failed;
                continue;
            }

            // Stamp metadata from dataset entry
            if constexpr (requires { entry.surface_id; }) {
                (*img_result)->Meta().surface_id = entry.surface_id;
                (*img_result)->Meta().position_id = entry.position_id;
                (*img_result)->Meta().light_id = entry.light_id;
            }

            FrameRecord rec;
            rec.frame_id = static_cast<int>(i);
            rec.image_path = entry.path.string();
            if constexpr (requires { entry.position_id; }) {
                rec.position_id = entry.position_id;
            } else {
                rec.position_id = 0;
            }
            if constexpr (requires { entry.light_id; }) {
                rec.light_id = entry.light_id;
            } else {
                rec.light_id = 0;
            }
            if constexpr (requires { entry.surface_id; }) {
                rec.surface_id = entry.surface_id;
            }
            if constexpr (requires { entry.expected_verdict; }) {
                rec.expected_verdict = entry.expected_verdict.value_or("");
            }

            auto* raw = dynamic_cast<image::RawImage*>(img_result->get());
            if (raw) {
                img_result->release();
                (void)app.pipeline->Submit(std::move(*raw));
            } else {
                auto meta = (*img_result)->Meta();
                const auto* data = (*img_result)->Data();
                auto size = (*img_result)->SizeBytes();
                std::vector<std::uint8_t> buffer(data, data + size);
                (void)app.pipeline->Submit(
                    image::RawImage::FromOwnedBuffer(std::move(buffer), meta));
            }

            (void)app.pipeline->Drain();

            // Read the exported JSON result
            auto result_path = std::filesystem::path(cli.output_dir)
                / "default" / "unknown" / "result.json";
            std::string verdict = "?";
            double severity = 0.0;
            double confidence = 0.0;
            if (std::filesystem::exists(result_path)) {
                std::ifstream ifs(result_path);
                try {
                    auto j = nlohmann::json::parse(ifs);
                    verdict = j.value("verdict", "?");
                    severity = j.value("severity", 0.0);
                    confidence = j.value("confidence", 0.0);
                } catch (const nlohmann::json::exception& e) {
                    std::cerr << "JSON parse error: " << e.what() << "\n";
                }
            }

            rec.verdict = verdict;
            rec.severity = severity;
            rec.confidence = confidence;
            records.push_back(std::move(rec));

            if (verdict == "OK") ++ok;
            else if (verdict == "NG") ++ng;
            else if (verdict == "WARN") ++warn;
            else if (verdict == "UNCERTAIN") ++uncertain;
            else ++failed;

            // Per-surface tracking: worst verdict wins
            if constexpr (requires { entry.surface_id; }) {
                auto& sv = surface_verdict[entry.surface_id];
                if (verdict == "NG" || (sv != "NG" && verdict == "WARN")
                    || (sv != "NG" && sv != "WARN" && verdict == "UNCERTAIN")
                    || (sv.empty() && verdict == "OK"))
                    sv = verdict;
                surface_frame_count[entry.surface_id]++;
            }

            std::cout << "[" << (i + 1) << "/" << entries.size() << "] "
                      << entry.path.filename().string() << " -> " << verdict << "\n";
        }

        std::cout << "\n===== Frame Summary =====\n"
                  << "Total:   " << entries.size() << "\n"
                  << "OK:      " << ok << "\n"
                  << "NG:      " << ng << "\n"
                  << "WARN:    " << warn << "\n"
                  << "UNCERTAIN: " << uncertain << "\n"
                  << "Failed:  " << failed << "\n";

        if (!surface_verdict.empty()) {
            int s_ok = 0, s_ng = 0, s_warn = 0, s_uncertain = 0;
            for (auto& [sid, v] : surface_verdict) {
                if (v == "OK") ++s_ok;
                else if (v == "NG") ++s_ng;
                else if (v == "WARN") ++s_warn;
                else if (v == "UNCERTAIN") ++s_uncertain;
                std::cout << "  [" << sid << "] " << v
                          << " (" << surface_frame_count[sid] << " frames)\n";
            }
            std::cout << "\n===== Product Summary =====\n"
                      << "Products: " << surface_verdict.size() << "\n"
                      << "OK:       " << s_ok << "\n"
                      << "NG:       " << s_ng << "\n"
                      << "WARN:     " << s_warn << "\n"
                      << "UNCERTAIN:" << s_uncertain << "\n";
        }

        // Metrics: confusion matrix + precision/recall/F1
        MetricsCounts metrics;
        for (auto& rec : records) {
            if (rec.expected_verdict.empty()) continue;  // no ground truth -> skip
            metrics.Add(rec.verdict, rec.expected_verdict);
        }
        int labeled = metrics.actual_ok + metrics.actual_ng;
        if (labeled > 0) {
            std::cout << "\n===== Detection Metrics =====\n"
                      << "Labeled frames: " << labeled << "\n"
                      << "  Actual OK:  " << metrics.actual_ok << "\n"
                      << "  Actual NG:  " << metrics.actual_ng << "\n\n"
                      << "Confusion Matrix (binary, WARN/UNCERTAIN excluded):\n"
                      << "              Predicted\n"
                      << "              OK    NG\n"
                      << "Actual OK     " << metrics.tp << "     " << metrics.fn << "\n"
                      << "Actual NG     " << metrics.fp << "     " << metrics.tn << "\n\n"
                      << "Precision: " << metrics.Precision() << "\n"
                      << "Recall:    " << metrics.Recall() << "\n"
                      << "F1 Score:  " << metrics.F1() << "\n"
                      << "Accuracy:  " << metrics.Accuracy() << "\n";
        }

        std::cout << "Results: " << cli.output_dir << "\n";

        // Write review_index.json
        if (!records.empty()) {
            std::string surface_id = records.front().surface_id;
            WriteReviewIndex(cli.output_dir, records, surface_id);
            std::cout << "Review index: " << cli.output_dir << "/review_index.json\n";
        }

        if (app.evolution != nullptr) app.evolution->Stop();
        for (auto& [key, evo] : app.evolutions) {
            evo->Stop();
        }
        if (app.tuning_scheduler != nullptr) app.tuning_scheduler->Join();
        (void)app.pipeline->Stop();
        (void)app.ctx->Stop();
        return (ng > 0 || failed > 0) ? 1 : 0;
    };

    if (!cli.dataset_path.empty()) {
        io::BasicImporter importer;
        auto entries_result = io::BasicImporter::ImportDataset(cli.dataset_path);
        if (!entries_result) {
            std::cerr << "Dataset import failed: "
                      << entries_result.error().message << "\n";
            return 1;
        }
        auto& entries = *entries_result;
        std::cout << "Processing " << entries.size() << " images from "
                  << cli.dataset_path << "\n"
                  << "Output: " << cli.output_dir << "\n\n";
        return process_entries(entries, importer);
    }

    if (!cli.image_dir.empty()) {
        io::BasicImporter importer;
        std::vector<std::filesystem::path> image_files;
        for (auto& entry : std::filesystem::directory_iterator(cli.image_dir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            if (ext == ".ppm" || ext == ".png" || ext == ".bmp"
                || ext == ".jpg" || ext == ".jpeg") {
                image_files.push_back(entry.path());
            }
        }
        std::sort(image_files.begin(), image_files.end());

        struct DirEntry {
            std::filesystem::path path;
        };
        std::vector<DirEntry> dir_entries;
        dir_entries.reserve(image_files.size());
        for (auto& p : image_files)
            dir_entries.push_back({std::move(p)});

        std::cout << "Processing " << dir_entries.size() << " images from "
                  << cli.image_dir << "\n"
                  << "Output: " << cli.output_dir << "\n\n";
        return process_entries(dir_entries, importer);
    }

    return 0;
}

void WriteReviewIndex(std::string_view output_dir,
                      const std::vector<FrameRecord>& records,
                      std::string_view surface_id) {
    using json = nlohmann::json;
    json index;
    index["surface_id"] = surface_id;
    index["total_frames"] = records.size();

    json frames_arr = json::array();
    for (auto& rec : records) {
        json f;
        f["frame_id"] = rec.frame_id;
        f["image_path"] = rec.image_path;
        f["verdict"] = rec.verdict;
        f["severity"] = rec.severity;
        f["confidence"] = rec.confidence;
        f["position_id"] = rec.position_id;
        f["light_id"] = rec.light_id;
        f["surface_id"] = rec.surface_id;
        if (!rec.expected_verdict.empty())
            f["expected_verdict"] = rec.expected_verdict;
        frames_arr.push_back(std::move(f));
    }
    index["frames"] = std::move(frames_arr);

    auto path = std::filesystem::path(output_dir) / "review_index.json";
    std::ofstream ofs(path);
    ofs << index.dump(2);  // pretty-print with 2-space indent
}
