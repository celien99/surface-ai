#include "headless_runner.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "app_builder.h"
#include "cli_args.h"

#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/reasoner/reasoner.h>

auto RunHeadless(const CliArgs& cli, AssembledApp& app) -> int {
    using namespace sai;

    auto process_entries = [&](auto& entries, io::BasicImporter& importer) -> int {
        int ok = 0, ng = 0, warn = 0, uncertain = 0, failed = 0;
        int frame_id = 0;

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
            ++frame_id;

            // Read the exported JSON result
            auto result_path = std::filesystem::path(cli.output_dir)
                / "default" / "unknown" / "result.json";
            std::string verdict = "?";
            if (std::filesystem::exists(result_path)) {
                std::ifstream ifs(result_path);
                std::string line;
                while (std::getline(ifs, line)) {
                    auto pos = line.find("\"verdict\"");
                    if (pos != std::string::npos) {
                        pos = line.find(':', pos);
                        if (pos != std::string::npos) {
                            auto start = line.find('"', pos);
                            auto end = line.find('"', start + 1);
                            if (start != std::string::npos && end != std::string::npos) {
                                verdict = line.substr(start + 1, end - start - 1);
                            }
                        }
                        break;
                    }
                }
            }

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
                      << entry.path.filename().string() << " → " << verdict << "\n";
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

        std::cout << "Results: " << cli.output_dir << "\n";

        if (app.evolution.has_value()) app.evolution->Stop();
        if (app.tuning_scheduler.has_value()) app.tuning_scheduler->Join();
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

        // Wrap paths in lightweight entries for the generic lambda
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
