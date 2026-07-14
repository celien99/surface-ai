#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <chrono>
#include <thread>

#include "sai/core/context.h"
#include "sai/device/fake_camera.h"
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

    // 2. Initialize + Start
    (void)ctx->Initialize();
    (void)ctx->Start();

    // 3. Load Pipeline from YAML
    auto pipeline_result = pipeline::Pipeline::LoadFromYAML(
        "resources/pipeline.yaml", *ctx);
    if (!pipeline_result) {
        qFatal("Pipeline load failed: %s",
               pipeline_result.error().message.c_str());
    }
    auto pipeline = std::move(*pipeline_result);

    // 4. Create ViewModels
    auto* pipeline_vm = new visualization::PipelineViewModel(&app);
    pipeline_vm->BindToPipeline(pipeline.get());

    auto* inspection_vm = new visualization::InspectionViewModel(&app);
    auto* config_vm = new visualization::ConfigViewModel(&app);
    auto* dashboard_vm = new visualization::DashboardViewModel(&app);
    auto* frame_provider = new visualization::FrameProvider();

    // 5. Wire result callback — invoked per frame from Export worker thread
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

    // 6. Start Pipeline (worker threads begin polling input queues)
    auto start_result = pipeline->Start();
    if (!start_result) {
        qFatal("Pipeline start failed: %s",
               start_result.error().message.c_str());
    }

    // 7. Create FakeCamera and wire it to drive the pipeline.
    //    FrameCallback calls Pipeline::Submit(), pushing frames into the
    //    CaptureStage input queue where worker threads pick them up.
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

    // 8. Register QML context properties
    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    // 9. Start ViewModel refresh
    pipeline_vm->StartRefresh(33);

    // 10. Load QML
    engine.load("qrc:/MainWindow.qml");

    // 11. Cleanup on exit
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
