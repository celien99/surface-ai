#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "sai/core/context.h"
#include "sai/pipeline/pipeline.h"
#include "sai/reasoner/reasoner.h"
#include "sai/visualization/pipeline_viewmodel.h"
#include "sai/visualization/inspection_viewmodel.h"
#include "sai/visualization/frame_provider.h"
#include "sai/visualization/config_viewmodel.h"
#include "sai/visualization/dashboard_viewmodel.h"

auto main(int argc, char* argv[]) -> int {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    using namespace sai;

    // 1. Create DI container
    auto ctx = std::make_unique<Context>();

    // 2. Initialize + Start (modules registered via Context in a full deployment)
    //    For M7: Pipeline::LoadFromYAML handles module assembly internally
    ctx->Initialize();
    (void)ctx->Start();

    // 3. Load Pipeline from YAML
    auto pipeline_result = pipeline::Pipeline::LoadFromYAML("resources/pipeline.yaml", *ctx);
    if (!pipeline_result) {
        qFatal("Pipeline load failed: %s", pipeline_result.error().message.c_str());
    }
    auto& pipeline = *pipeline_result;

    // 4. Create ViewModels
    auto* pipeline_vm = new visualization::PipelineViewModel(&app);
    pipeline_vm->BindToPipeline(pipeline.get());

    auto* inspection_vm = new visualization::InspectionViewModel(&app);
    auto* config_vm = new visualization::ConfigViewModel(&app);
    auto* dashboard_vm = new visualization::DashboardViewModel(&app);

    auto* frame_provider = new visualization::FrameProvider();

    // 5. Wire result callback
    pipeline->SetResultCallback([=](int frame_id, const reasoner::ReasoningResult& result) {
        inspection_vm->UpdateResult(frame_id, result);

        visualization::FrameSummary summary;
        summary.frame_id = frame_id;
        summary.verdict = result.verdict;
        summary.severity = std::to_string(result.severity);
        summary.timestamp = std::chrono::system_clock::now();
        dashboard_vm->AppendFrameSummary(std::move(summary));
    });

    // 6. Start Pipeline
    auto start_result = pipeline->Start();
    if (!start_result) {
        qFatal("Pipeline start failed: %s", start_result.error().message.c_str());
    }

    // 7. Register QML context properties
    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    // 8. Start ViewModel refresh
    pipeline_vm->StartRefresh(33);

    // 9. Load QML
    engine.load("qrc:/MainWindow.qml");

    // 10. Cleanup on exit
    QObject::connect(&app, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
        (void)pipeline->Drain();
        (void)pipeline->Stop();
        (void)ctx->Stop();
    });

    return app.exec();
}
