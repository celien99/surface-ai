#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sqlite3.h>

#include "sai/core/context.h"
#include "pipeline/stage_nodes.h"
#include "sai/device/fake_camera.h"
#include "sai/image/raw_image.h"
#include "sai/io/importer.h"
#include "sai/io/exporter.h"

// Inference: TensorRT on Linux+CUDA, MockEngine elsewhere
#if defined(__linux__)
#include <sai/inference/tensorrt_engine.h>
#else
#include <sai/inference/mock_engine.h>
#endif

#include <sai/detection/patch_core.h>
#include <sai/detection/feature_bank.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>
#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>
#include <sai/reasoner/decision_tree.h>
#include <sai/pipeline/pipeline.h>
#include <sai/visualization/pipeline_viewmodel.h>
#include <sai/visualization/inspection_viewmodel.h>
#include <sai/visualization/frame_provider.h>
#include <sai/visualization/config_viewmodel.h>
#include <sai/visualization/dashboard_viewmodel.h>

namespace {

struct CliArgs {
    std::string image_dir;
    std::string output_dir = "/tmp/surface-ai/results/";
    bool headless = false;
};

auto ParseArgs(int argc, char* argv[]) -> CliArgs {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--image-dir" && i + 1 < argc) {
            args.image_dir = argv[++i];
            args.headless = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--headless") {
            args.headless = true;
        }
    }
    return args;
}

// Populate KnowledgeGraph with sample seat leather inspection domain data.
// In production, this comes from MES/PLC via OPC UA.
auto SeedKnowledgeGraph(sai::knowledge::KnowledgeGraph& kg) -> void {
    using namespace sai::knowledge;

    auto make_record = [](std::initializer_list<std::pair<const char*, FieldValue>> items) {
        KnowledgeRecord r;
        for (auto& [k, v] : items) r.fields[k] = v;
        return r;
    };

    // --- Material nodes ---
    auto mat = kg.InsertNode("Material",
        make_record({{"name", std::string("Nappa_Leather_Black")},
                     {"sku", std::string("SKU-NLB-001")},
                     {"thickness_mm", 1.2}}));
    auto mat2 = kg.InsertNode("Material",
        make_record({{"name", std::string("Nappa_Leather_Brown")},
                     {"sku", std::string("SKU-NLB-002")},
                     {"thickness_mm", 1.2}}));

    // --- Supplier nodes ---
    auto supp = kg.InsertNode("Supplier",
        make_record({{"name", std::string("LeatherWorks_GmbH")},
                     {"location", std::string("Stuttgart_DE")},
                     {"certification", std::string("ISO_9001")}}));
    auto supp2 = kg.InsertNode("Supplier",
        make_record({{"name", std::string("PremiumHide_Ltd")},
                     {"location", std::string("Modena_IT")},
                     {"certification", std::string("ISO_9001")}}));

    // --- Batch nodes ---
    auto batch1 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-001")},
                     {"reject_rate_pct", 0.8},
                     {"total_units", static_cast<std::int64_t>(5000)},
                     {"surface", std::string("seat_leather_driver")}}));
    auto batch2 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-002")},
                     {"reject_rate_pct", 4.2},
                     {"total_units", static_cast<std::int64_t>(3200)},
                     {"surface", std::string("seat_leather_passenger")}}));
    auto batch3 = kg.InsertNode("Batch",
        make_record({{"batch_id", std::string("B2026-003")},
                     {"reject_rate_pct", 1.5},
                     {"total_units", static_cast<std::int64_t>(8000)},
                     {"surface", std::string("seat_leather_driver")}}));

    // --- Edges ---
    (void)kg.InsertEdge(supp.value(), mat.value(), "SUPPLIES",
        make_record({{"lead_time_days", static_cast<std::int64_t>(14)}}));
    (void)kg.InsertEdge(supp2.value(), mat2.value(), "SUPPLIES",
        make_record({{"lead_time_days", static_cast<std::int64_t>(21)}}));
    (void)kg.InsertEdge(mat.value(), batch1.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L3")}}));
    (void)kg.InsertEdge(mat.value(), batch3.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L3")}}));
    (void)kg.InsertEdge(mat2.value(), batch2.value(), "PRODUCED_AS",
        make_record({{"production_line", std::string("L5")}}));
}

}  // anonymous namespace

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    using namespace sai;

    // =========================================================================
    // 1. DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // 2. Inference Engine — TensorRT on Linux+CUDA, MockEngine on macOS
    // =========================================================================
#if defined(__linux__)
    auto infer_engine = std::make_shared<inference::TensorRtEngine>(/*device_ordinal=*/0);
    std::cout << "Inference: TensorRtEngine (GPU)\n";
#else
    auto infer_engine = std::make_shared<inference::MockEngine>();
    std::cout << "Inference: MockEngine (no GPU on this host)\n";
#endif

    // =========================================================================
    // 3. Detection — PatchCore with DINOv3-compatible config
    // =========================================================================
    detection::PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = 1024;            // DINOv3 ViT-B/14 output dim
    pc_cfg.image_width = 1024;
    pc_cfg.image_height = 1024;
    pc_cfg.patch_size = 14;
    pc_cfg.k_nearest = 5;
    pc_cfg.anomaly_threshold = 0.8F;
    pc_cfg.enable_adaptive_threshold = true;
    pc_cfg.target_fpr = 0.01F;
    auto patch_core = std::make_shared<detection::PatchCore>(std::move(pc_cfg));
    (void)patch_core->Initialize(*ctx);

    // =========================================================================
    // 4. FeatureBank + VectorPath — FAISS vector search
    //    FeatureBank::LoadFromFile(coreset_path, dim) loads pre-computed
    //    coreset embeddings. VectorPath wraps it for FAISS TopK/Range search.
    //    On first run without a coreset file, VectorPath is unavailable;
    //    RuleEvalStage falls back to bare detection facts (no retrieval).
    //    To enable: provide --coreset /path/to/coreset.bin after training.
    // =========================================================================
    std::shared_ptr<detection::FeatureBank> feature_bank;
    std::shared_ptr<retrieval::VectorPath> vp;

    // =========================================================================
    // 5. KnowledgeGraph — in-memory SQLite with sample seat leather data
    // =========================================================================
    sqlite3* sqlite_db = nullptr;
    sqlite3_open(":memory:", &sqlite_db);
    auto kg = std::make_shared<knowledge::KnowledgeGraph>(sqlite_db);
    SeedKnowledgeGraph(*kg);

    // =========================================================================
    // 6. Rule Engine + Reasoner — loaded from YAML
    // =========================================================================
    auto rule_engine = std::make_shared<rule::RuleEngine>();

    auto tree_result = reasoner::DecisionTree::LoadFromYAML(
        "resources/trees/seat_leather_inspection.yaml");
    std::shared_ptr<reasoner::IReasoner> reasoner;
    if (tree_result) {
        reasoner = std::make_shared<reasoner::DefaultReasoner>(
            std::move(*tree_result));
    } else {
        std::cerr << "Warning: decision tree load failed, reasoner unavailable\n";
    }

    // =========================================================================
    // 7. Export
    // =========================================================================
    auto exporter = std::make_shared<io::JsonExporter>();

    // =========================================================================
    // 8. Load Pipeline from YAML
    // =========================================================================
    auto pipeline_result = pipeline::Pipeline::LoadFromYAML(
        "resources/pipeline.yaml", *ctx);
    if (!pipeline_result) {
        std::cerr << "Pipeline load failed: "
                  << pipeline_result.error().message << "\n";
        return 1;
    }
    auto pipeline = std::move(*pipeline_result);

    // =========================================================================
    // 9. Wire all business objects into Pipeline stages
    // =========================================================================
    if (auto* s = pipeline->GetStage("inference"))
        static_cast<pipeline::InferenceStage*>(s)->SetEngine(infer_engine);
    if (auto* s = pipeline->GetStage("detect"))
        static_cast<pipeline::DetectStage*>(s)->SetDetector(patch_core);
    if (auto* s = pipeline->GetStage("rule_eval")) {
        auto* rs = static_cast<pipeline::RuleEvalStage*>(s);
        rs->SetRuleEngine(rule_engine);
        rs->SetKnowledgeGraph(kg);
        if (vp) rs->SetVectorPath(vp);
    }
    if (auto* s = pipeline->GetStage("reason"))
        static_cast<pipeline::ReasonStage*>(s)->SetReasoner(reasoner);
    if (auto* s = pipeline->GetStage("export"))
        static_cast<pipeline::ExportStage*>(s)->SetExporter(exporter);

    // =========================================================================
    // 10. Start Pipeline
    // =========================================================================
    auto start_result = pipeline->Start();
    if (!start_result) {
        std::cerr << "Pipeline start failed: "
                  << start_result.error().message << "\n";
        return 1;
    }

    // =========================================================================
    // 11. Headless batch mode — process images from directory
    // =========================================================================
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

        std::cout << "Processing " << image_files.size() << " images from "
                  << cli.image_dir << "\n"
                  << "Output: " << cli.output_dir << "\n\n";

        int ok = 0, ng = 0, warn = 0, uncertain = 0, failed = 0;
        int frame_id = 0;
        for (size_t i = 0; i < image_files.size(); ++i) {
            auto& path = image_files[i];
            auto img_result = importer.ImportImage(path);
            if (!img_result) {
                std::cerr << "FAIL import: " << path.filename() << "\n";
                ++failed;
                continue;
            }

            auto* raw = dynamic_cast<image::RawImage*>(img_result->get());
            if (raw) {
                img_result->release();
                (void)pipeline->Submit(std::move(*raw));
            } else {
                auto meta = (*img_result)->Meta();
                const auto* data = (*img_result)->Data();
                auto size = (*img_result)->SizeBytes();
                std::vector<std::uint8_t> buffer(data, data + size);
                (void)pipeline->Submit(
                    image::RawImage::FromOwnedBuffer(std::move(buffer), meta));
            }

            (void)pipeline->Drain();
            ++frame_id;

            // Read the exported JSON result
            auto result_path = std::filesystem::path(cli.output_dir)
                / "default" / "unknown" / "result.json";
            std::string verdict = "?";
            if (std::filesystem::exists(result_path)) {
                // Quick parse: check verdict field
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

            std::cout << "[" << (i + 1) << "/" << image_files.size() << "] "
                      << path.filename().string() << " → " << verdict << "\n";
        }

        std::cout << "\n===== Summary =====\n"
                  << "Total:   " << image_files.size() << "\n"
                  << "OK:      " << ok << "\n"
                  << "NG:      " << ng << "\n"
                  << "WARN:    " << warn << "\n"
                  << "UNCERTAIN: " << uncertain << "\n"
                  << "Failed:  " << failed << "\n"
                  << "Results: " << cli.output_dir << "\n";

        (void)pipeline->Stop();
        (void)ctx->Stop();
        return (ng > 0 || failed > 0) ? 1 : 0;
    }

    // =========================================================================
    // 12. GUI mode — FakeCamera drives the frame loop
    // =========================================================================
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    auto* pipeline_vm = new visualization::PipelineViewModel(&app);
    pipeline_vm->BindToPipeline(pipeline.get());
    auto* inspection_vm = new visualization::InspectionViewModel(&app);
    auto* config_vm = new visualization::ConfigViewModel(&app);
    auto* dashboard_vm = new visualization::DashboardViewModel(&app);
    auto* frame_provider = new visualization::FrameProvider();

    pipeline->SetResultCallback(
        [=](int fid, const reasoner::ReasoningResult& result) {
            inspection_vm->UpdateResult(fid, result);
            visualization::FrameSummary summary;
            summary.frame_id = fid;
            summary.verdict = result.verdict;
            summary.severity = std::to_string(result.severity);
            summary.timestamp = std::chrono::system_clock::now();
            dashboard_vm->AppendFrameSummary(std::move(summary));
        });

    device::FakeCamera::Config cam_cfg{
        .width = 1024, .height = 1024, .fps = 10.0};
    auto camera = std::make_shared<device::FakeCamera>(cam_cfg);
    auto* pipeline_ptr = pipeline.get();
    (void)camera->RegisterFrameCallback(
        [pipeline_ptr](sai::image::RawImage img) {
            (void)pipeline_ptr->Submit(std::move(img));
        });
    (void)camera->Connect();
    (void)camera->SetTriggerMode(device::ICamera::TriggerMode::FreeRun);
    (void)camera->StartAcquisition();

    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);
    pipeline_vm->StartRefresh(33);
    engine.load("qrc:/MainWindow.qml");

    QObject::connect(&app, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
        (void)camera->StopAcquisition();
        (void)camera->Disconnect();
        (void)pipeline->Drain();
        (void)pipeline->Stop();
        (void)ctx->Stop();
    });

    return app.exec();
}
