#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sqlite3.h>

#include "sai/core/context.h"
#include "pipeline/stage_nodes.h"  // internal header: concrete stage types for DI wiring
#include "sai/device/fake_camera.h"
#include "sai/image/raw_image.h"
#include "sai/io/importer.h"
#include "sai/inference/mock_engine.h"
#include "sai/detection/patch_core.h"
#include "sai/knowledge/knowledge_graph.h"
#include "sai/rule/rule_engine.h"
#include "sai/reasoner/reasoner.h"
#include "sai/reasoner/decision_tree.h"
#include "sai/io/exporter.h"
#include "sai/pipeline/pipeline.h"
#include "sai/visualization/pipeline_viewmodel.h"
#include "sai/visualization/inspection_viewmodel.h"
#include "sai/visualization/frame_provider.h"
#include "sai/visualization/config_viewmodel.h"
#include "sai/visualization/dashboard_viewmodel.h"

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

}  // anonymous namespace

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    using namespace sai;

    // =========================================================================
    // 1. Create DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // 2. Create business objects
    // =========================================================================
    auto mock_engine = std::make_shared<inference::MockEngine>();

    auto patch_core = std::make_shared<detection::PatchCore>(
        detection::PatchCore::Config{});
    (void)patch_core->Initialize(*ctx);

    auto rule_engine = std::make_shared<rule::RuleEngine>();

    // KnowledgeGraph: in-memory SQLite property graph for material/batch data.
    // VectorPath requires FeatureBank (real embeddings), deferred to GPU target.
    sqlite3* sqlite_db = nullptr;
    sqlite3_open(":memory:", &sqlite_db);
    auto kg = std::make_shared<knowledge::KnowledgeGraph>(sqlite_db);

    auto tree_result = reasoner::DecisionTree::LoadFromYAML(
        "resources/trees/seat_leather_inspection.yaml");
    std::shared_ptr<reasoner::IReasoner> reasoner;
    if (tree_result) {
        reasoner = std::make_shared<reasoner::DefaultReasoner>(
            std::move(*tree_result));
    }

    auto exporter = std::make_shared<io::JsonExporter>();

    // =========================================================================
    // 3. Load Pipeline
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
    // 4. Wire business objects
    // =========================================================================
    if (auto* s = pipeline->GetStage("inference"))
        static_cast<pipeline::InferenceStage*>(s)->SetEngine(mock_engine);
    if (auto* s = pipeline->GetStage("detect"))
        static_cast<pipeline::DetectStage*>(s)->SetDetector(patch_core);
    if (auto* s = pipeline->GetStage("rule_eval")) {
        auto* rs = static_cast<pipeline::RuleEvalStage*>(s);
        rs->SetRuleEngine(rule_engine);
        rs->SetKnowledgeGraph(kg);
    }
    if (auto* s = pipeline->GetStage("reason"))
        static_cast<pipeline::ReasonStage*>(s)->SetReasoner(reasoner);
    if (auto* s = pipeline->GetStage("export"))
        static_cast<pipeline::ExportStage*>(s)->SetExporter(exporter);

    // =========================================================================
    // 5. Start Pipeline
    // =========================================================================
    auto start_result = pipeline->Start();
    if (!start_result) {
        std::cerr << "Pipeline start failed: "
                  << start_result.error().message << "\n";
        return 1;
    }

    // =========================================================================
    // 6. Headless batch mode: process images from directory
    // =========================================================================
    if (!cli.image_dir.empty()) {
        io::BasicImporter importer;
        std::vector<std::filesystem::path> image_files;

        // Collect image files sorted by name
        for (auto& entry : std::filesystem::directory_iterator(cli.image_dir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                if (ext == ".ppm" || ext == ".png" || ext == ".bmp"
                    || ext == ".jpg" || ext == ".jpeg") {
                    image_files.push_back(entry.path());
                }
            }
        }
        std::sort(image_files.begin(), image_files.end());

        std::cout << "Processing " << image_files.size() << " images from "
                  << cli.image_dir << "\n";

        int ok_count = 0, ng_count = 0;
        for (size_t i = 0; i < image_files.size(); ++i) {
            auto& path = image_files[i];
            auto img_result = importer.ImportImage(path);
            if (!img_result) {
                std::cerr << "Failed to import: " << path << "\n";
                continue;
            }

            // ImportImage returns unique_ptr<Image>. If it's a RawImage,
            // pass directly; otherwise wrap buffer.
            auto* raw = dynamic_cast<image::RawImage*>(img_result->get());
            if (raw) {
                img_result->release();
                (void)pipeline->Submit(std::move(*raw));
            } else {
                // Wrap as RawImage from buffer
                auto meta = (*img_result)->Meta();
                const auto* data = (*img_result)->Data();
                auto size = (*img_result)->SizeBytes();
                std::vector<std::uint8_t> buffer(data, data + size);
                (void)pipeline->Submit(
                    image::RawImage::FromOwnedBuffer(std::move(buffer), meta));
            }

            // Drain after each frame to wait for pipeline completion
            (void)pipeline->Drain();

            std::cout << "[" << (i + 1) << "/" << image_files.size() << "] "
                      << path.filename().string() << "\n";
        }

        (void)pipeline->Drain();
        (void)pipeline->Stop();
        (void)ctx->Stop();
        std::cout << "Done. Results in " << cli.output_dir << "\n";
        return 0;
    }

    // =========================================================================
    // 7. GUI mode: FakeCamera drives the frame loop
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
        [=](int frame_id, const reasoner::ReasoningResult& result) {
            inspection_vm->UpdateResult(frame_id, result);
            visualization::FrameSummary summary;
            summary.frame_id = frame_id;
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
