#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <chrono>
#include <memory>

#include "sai/core/context.h"
#include "pipeline/stage_nodes.h"  // internal header: concrete stage types for DI wiring
#include "sai/device/fake_camera.h"
#include "sai/inference/mock_engine.h"
#include "sai/detection/patch_core.h"
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

auto main(int argc, char* argv[]) -> int {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    using namespace sai;

    // =========================================================================
    // 1. Create DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // 2. Create business objects (owned by main, wired via setters)
    // =========================================================================

    // 2a. Inference: MockEngine (no-op, returns empty DetectionResult)
    auto mock_engine = std::make_shared<inference::MockEngine>();

    // 2b. Detection: PatchCore with minimal config
    auto patch_core = std::make_shared<detection::PatchCore>(
        detection::PatchCore::Config{});
    (void)patch_core->Initialize(*ctx);

    // 2c. Rule Engine: loads rules from YAML (wired via setter)
    auto rule_engine = std::make_shared<rule::RuleEngine>();

    // 2d. Reasoner: DefaultReasoner with decision tree from YAML
    auto tree_result = reasoner::DecisionTree::LoadFromYAML(
        "resources/trees/seat_leather_inspection.yaml");
    std::shared_ptr<reasoner::IReasoner> reasoner;
    if (tree_result) {
        reasoner = std::make_shared<reasoner::DefaultReasoner>(
            std::move(*tree_result));
    }

    // 2e. Export: JsonExporter (stateless, writes JSON + PPM to disk)
    auto exporter = std::make_shared<io::JsonExporter>();

    // =========================================================================
    // 3. Load Pipeline from YAML
    // =========================================================================
    auto pipeline_result = pipeline::Pipeline::LoadFromYAML(
        "resources/pipeline.yaml", *ctx);
    if (!pipeline_result) {
        qFatal("Pipeline load failed: %s",
               pipeline_result.error().message.c_str());
    }
    auto pipeline = std::move(*pipeline_result);

    // =========================================================================
    // 4. Wire business objects into Pipeline stages via setters
    // =========================================================================
    if (auto* s = pipeline->GetStage("inference")) {
        static_cast<pipeline::InferenceStage*>(s)->SetEngine(mock_engine);
    }
    if (auto* s = pipeline->GetStage("detect")) {
        static_cast<pipeline::DetectStage*>(s)->SetDetector(patch_core);
    }
    if (auto* s = pipeline->GetStage("rule_eval")) {
        static_cast<pipeline::RuleEvalStage*>(s)->SetRuleEngine(rule_engine);
    }
    if (auto* s = pipeline->GetStage("reason")) {
        static_cast<pipeline::ReasonStage*>(s)->SetReasoner(reasoner);
    }
    if (auto* s = pipeline->GetStage("export")) {
        static_cast<pipeline::ExportStage*>(s)->SetExporter(exporter);
    }

    // =========================================================================
    // 5. Create ViewModels
    // =========================================================================
    auto* pipeline_vm = new visualization::PipelineViewModel(&app);
    pipeline_vm->BindToPipeline(pipeline.get());

    auto* inspection_vm = new visualization::InspectionViewModel(&app);
    auto* config_vm = new visualization::ConfigViewModel(&app);
    auto* dashboard_vm = new visualization::DashboardViewModel(&app);
    auto* frame_provider = new visualization::FrameProvider();

    // =========================================================================
    // 6. Wire result callback
    // =========================================================================
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

    // =========================================================================
    // 7. Start Pipeline (worker threads begin polling)
    // =========================================================================
    auto start_result = pipeline->Start();
    if (!start_result) {
        qFatal("Pipeline start failed: %s",
               start_result.error().message.c_str());
    }

    // =========================================================================
    // 8. FakeCamera — drives the frame loop via callback → Pipeline::Submit
    // =========================================================================
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

    // =========================================================================
    // 9. QML context
    // =========================================================================
    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    pipeline_vm->StartRefresh(33);
    engine.load("qrc:/MainWindow.qml");

    // =========================================================================
    // 10. Cleanup
    // =========================================================================
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
