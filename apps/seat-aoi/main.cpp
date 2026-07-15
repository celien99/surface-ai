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
#include "sai/image/surface_image.h"
#include "sai/image/preprocess.h"
#include "sai/io/importer.h"
#include "sai/io/exporter.h"

// Inference: TensorRT on Linux+CUDA, MockEngine elsewhere
#if defined(__linux__)
#include <sai/inference/tensorrt_engine.h>
#else
#include <sai/inference/mock_engine.h>
#endif

#include <sai/embedding/simple_patch_embedder.h>
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
    std::string coreset_path;           // --coreset: load pre-built coreset for detection
    std::string build_coreset_dir;      // --build-coreset: dir of normal images for coreset building
    std::string coreset_output_path;     // --coreset-output: where to save the built coreset
    std::string coreset_algo = "greedy"; // --coreset-algo: greedy | uniform
    std::size_t coreset_max_samples = 10000; // --coreset-max-samples N
    bool headless = false;
    bool cpu_mode = false;              // --cpu: force CPU embedder (no GPU)
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
        } else if (arg == "--coreset" && i + 1 < argc) {
            args.coreset_path = argv[++i];
        } else if (arg == "--build-coreset" && i + 1 < argc) {
            args.build_coreset_dir = argv[++i];
        } else if (arg == "--coreset-output" && i + 1 < argc) {
            args.coreset_output_path = argv[++i];
        } else if (arg == "--coreset-algo" && i + 1 < argc) {
            args.coreset_algo = argv[++i];
        } else if (arg == "--coreset-max-samples" && i + 1 < argc) {
            args.coreset_max_samples = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--cpu") {
            args.cpu_mode = true;
        }
    }
    if (!args.build_coreset_dir.empty() && args.coreset_output_path.empty()) {
        args.coreset_output_path = "resources/coreset.bin";
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

// Build coreset from normal (defect-free) sample images.
// Uses SimplePatchEmbedder for portable CPU feature extraction.
auto BuildCoreset(const CliArgs& cli) -> int {
    using namespace sai;

    std::cout << "Building coreset from normal samples in: "
              << cli.build_coreset_dir << "\n";

    io::BasicImporter importer;
    embedding::SimplePatchEmbedderConfig sp_cfg;
    sp_cfg.image_width = 1024;
    sp_cfg.image_height = 1024;
    sp_cfg.patch_size = 14;
    sp_cfg.feature_dim = 128;

    auto emb_result = embedding::SimplePatchEmbedder::Create(sp_cfg);
    if (!emb_result) {
        std::cerr << "Failed to create SimplePatchEmbedder: "
                  << emb_result.error().message << "\n";
        return 1;
    }
    auto embedder = std::make_unique<embedding::SimplePatchEmbedder>(
        std::move(*emb_result));

    // Scan directory for image files
    std::vector<std::filesystem::path> image_files;
    for (auto& entry : std::filesystem::directory_iterator(cli.build_coreset_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext == ".ppm" || ext == ".png" || ext == ".bmp"
            || ext == ".jpg" || ext == ".jpeg") {
            image_files.push_back(entry.path());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cerr << "No image files found in " << cli.build_coreset_dir << "\n";
        return 1;
    }

    std::cout << "Found " << image_files.size() << " normal sample images\n";

    // Preprocess chain: resize to embedder's expected size
    auto preprocess_chain = image::Compose({
        image::MakeResize(sp_cfg.image_width, sp_cfg.image_height)
    });

    // Extract embeddings from each normal image
    std::vector<embedding::Embedding> all_embeddings;
    int processed = 0;
    for (auto& path : image_files) {
        auto img_result = importer.ImportImage(path);
        if (!img_result) {
            std::cerr << "Failed to import " << path.filename().string()
                      << ": " << img_result.error().message << "\n";
            continue;
        }

        // Run through preprocess (resize) to get SurfaceImage
        auto preprocess_result = preprocess_chain(std::move(*img_result));
        if (!preprocess_result) {
            std::cerr << "Preprocess failed for " << path.filename().string()
                      << ": " << preprocess_result.error().message << "\n";
            continue;
        }

        auto* surf_img = dynamic_cast<image::SurfaceImage*>(preprocess_result->get());
        if (surf_img == nullptr) {
            // Wrap as SurfaceImage
            auto meta = (*preprocess_result)->Meta();
            const auto* data = (*preprocess_result)->Data();
            auto size = (*preprocess_result)->SizeBytes();
            std::vector<std::uint8_t> buffer(data, data + size);
            auto surface = image::SurfaceImage::FromOwnedBuffer(std::move(buffer), meta);
            auto emb = embedder->Extract(surface);
            if (!emb) {
                std::cerr << "Embedding extraction failed for "
                          << path.filename().string() << "\n";
                continue;
            }
            all_embeddings.push_back(std::move(*emb));
        } else {
            auto emb = embedder->Extract(*surf_img);
            if (!emb) {
                std::cerr << "Embedding extraction failed for "
                          << path.filename().string() << "\n";
                continue;
            }
            all_embeddings.push_back(std::move(*emb));
        }

        ++processed;
        std::cout << "[" << processed << "/" << image_files.size() << "] "
                  << path.filename().string() << " → "
                  << all_embeddings.back().Meta().count << " patches\n";
    }

    if (all_embeddings.empty()) {
        std::cerr << "No embeddings extracted — coreset build failed\n";
        return 1;
    }

    // Build FeatureBank from all embeddings
    std::vector<const embedding::Embedding*> emb_ptrs;
    emb_ptrs.reserve(all_embeddings.size());
    for (auto& e : all_embeddings) emb_ptrs.push_back(&e);

    Result<detection::FeatureBank> bank_result = tl::make_unexpected(ErrorInfo{
        ErrorCode::Detection_FeatureBankLoadFailed,
        "No algorithm selected"});

    bool use_greedy = (cli.coreset_algo == "greedy");
    std::cout << "Coreset algorithm: " << (use_greedy ? "greedy (furthest-point sampling)" : "uniform")
              << "\nMax samples: " << cli.coreset_max_samples << "\n";

    if (use_greedy) {
        bank_result = detection::FeatureBank::BuildWithGreedyCoreset(
            emb_ptrs, sp_cfg.feature_dim, cli.coreset_max_samples);
    } else {
        bank_result = detection::FeatureBank::BuildFromEmbeddings(
            emb_ptrs, sp_cfg.feature_dim, cli.coreset_max_samples);
    }

    if (!bank_result) {
        std::cerr << "FeatureBank build failed: "
                  << bank_result.error().message << "\n";
        return 1;
    }

    // Save to file
    auto save_result = bank_result->SaveToFile(cli.coreset_output_path);
    if (!save_result) {
        std::cerr << "Failed to save coreset: "
                  << save_result.error().message << "\n";
        return 1;
    }

    // Output coverage statistics for greedy mode
    std::cout << "\nCoreset built successfully:\n"
              << "  Algorithm:    " << cli.coreset_algo << "\n"
              << "  Images:       " << processed << "\n"
              << "  Total patches:" << all_embeddings.size() * sp_cfg.feature_dim << "\n"
              << "  Coreset size: " << bank_result->NumSamples() << "\n"
              << "  Feature dim:  " << bank_result->Dim() << "\n"
              << "  Output:       " << cli.coreset_output_path << "\n";
    return 0;
}

}  // anonymous namespace

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    using namespace sai;

    // =========================================================================
    // --build-coreset mode: build FeatureBank from normal samples
    // =========================================================================
    if (!cli.build_coreset_dir.empty()) {
        return BuildCoreset(cli);
    }

    // =========================================================================
    // 1. DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // 2. Platform-dependent embed_dim and inference/embedding setup.
    //    CPU path (macOS): SimplePatchEmbedder with feature_dim=128
    //    GPU path (Linux): TensorRT + DINOv3 → PatchEmbedder with embed_dim=1024
    // =========================================================================
#if defined(__linux__)
    constexpr std::size_t kEmbedDim = 1024;
#else
    constexpr std::size_t kEmbedDim = 128;
#endif

    std::shared_ptr<embedding::IEmbedder> embedder;

#if defined(__linux__)
    // GPU path: TensorRT + DINOv3 adapter → PatchEmbedder
    {
        auto infer_engine = std::make_shared<inference::TensorRtEngine>(/*device_ordinal=*/0);
        std::cout << "Inference: TensorRtEngine (GPU)\n";

        inference::DinoV3Config dino_cfg;
        dino_cfg.engine_path = "resources/models/dino_v3_vit_base.engine";
        dino_cfg.image_size = 1024;
        dino_cfg.patch_size = 14;
        dino_cfg.embed_dim = kEmbedDim;

        auto dino_adapter = inference::DinoV3Adapter::Create(*infer_engine, dino_cfg);
        if (!dino_adapter) {
            std::cerr << "DinoV3Adapter creation failed: "
                      << dino_adapter.error().message << "\n";
            std::cerr << "Falling back to SimplePatchEmbedder (CPU)\n";
            // Fall through to CPU path below
        } else {
            auto patch_emb = embedding::PatchEmbedder::Create(std::move(*dino_adapter));
            if (patch_emb) {
                embedder = std::make_shared<embedding::PatchEmbedder>(std::move(*patch_emb));
                std::cout << "Embedder: PatchEmbedder (DINOv3)\n";
            }
        }
    }
#endif

    // CPU fallback (or primary path on macOS)
    if (!embedder) {
        embedding::SimplePatchEmbedderConfig sp_cfg;
        sp_cfg.image_width = 1024;
        sp_cfg.image_height = 1024;
        sp_cfg.patch_size = 14;
        sp_cfg.feature_dim = kEmbedDim;

        auto sp_result = embedding::SimplePatchEmbedder::Create(sp_cfg);
        if (!sp_result) {
            std::cerr << "SimplePatchEmbedder creation failed: "
                      << sp_result.error().message << "\n";
            return 1;
        }
        embedder = std::make_shared<embedding::SimplePatchEmbedder>(std::move(*sp_result));
        std::cout << "Embedder: SimplePatchEmbedder (CPU, dim=" << kEmbedDim << ")\n";
    }

    // =========================================================================
    // 3. Detection — PatchCore with matched embed_dim
    // =========================================================================
    detection::PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = kEmbedDim;
    pc_cfg.image_width = 1024;
    pc_cfg.image_height = 1024;
    pc_cfg.patch_size = 14;
    pc_cfg.k_nearest = 5;
    pc_cfg.anomaly_threshold = 0.8F;
    pc_cfg.enable_adaptive_threshold = true;
    pc_cfg.target_fpr = 0.01F;

    auto patch_core = std::make_shared<detection::PatchCore>(pc_cfg);

    // =========================================================================
    // 4. FeatureBank + VectorPath — load coreset if provided
    // =========================================================================
    std::shared_ptr<detection::FeatureBank> feature_bank;
    std::shared_ptr<retrieval::VectorPath> vp;

    if (!cli.coreset_path.empty()) {
        // Set the config path so PatchCore::Initialize can load it internally.
        pc_cfg.feature_bank_path = cli.coreset_path;

        // Also load separately for VectorPath use.
        auto fb_result = detection::FeatureBank::LoadFromFile(cli.coreset_path, kEmbedDim);
        if (fb_result) {
            feature_bank = std::make_shared<detection::FeatureBank>(std::move(*fb_result));
            std::cout << "FeatureBank loaded: " << feature_bank->NumSamples()
                      << " samples, dim=" << feature_bank->Dim() << "\n";

            vp = std::make_shared<retrieval::VectorPath>(*feature_bank);
            std::cout << "VectorPath: ready\n";
        } else {
            std::cerr << "Warning: failed to load coreset from "
                      << cli.coreset_path << ": "
                      << fb_result.error().message << "\n";
        }
    }

    if (!feature_bank) {
        std::cout << "FeatureBank: not loaded (detection will use PCA-only mode)\n";
    }

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
        static_cast<pipeline::InferenceStage*>(s)->SetEmbedder(embedder);
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
