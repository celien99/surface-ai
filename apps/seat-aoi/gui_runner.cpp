#include "gui_runner.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "app_builder.h"
#include "evolution_offer.h"

#include <sai/device/fake_camera.h>
#include <sai/image/raw_image.h>
#include <sai/reasoner/reasoner.h>
#include <sai/visualization/pipeline_viewmodel.h>
#include <sai/visualization/inspection_viewmodel.h>
#include <sai/visualization/frame_provider.h>
#include <sai/visualization/config_viewmodel.h>
#include <sai/visualization/dashboard_viewmodel.h>
#include <sai/detection/patch_core.h>
#include <sai/detection/coreset_evolution.h>
#include <sai/tuning/tuning_scheduler.h>

auto RunGui(int argc, char* argv[], AssembledApp& app) -> int {
    using namespace sai;

    QGuiApplication qapp(argc, argv);
    QQmlApplicationEngine engine;

    auto* pipeline_vm = new visualization::PipelineViewModel(&qapp);
    pipeline_vm->BindToPipeline(app.pipeline.get());
    auto* inspection_vm = new visualization::InspectionViewModel(&qapp);
    auto* config_vm = new visualization::ConfigViewModel(&qapp);
    config_vm->BindToPipeline(app.pipeline.get());
    config_vm->BindToRuleEngine(app.rule_engine.get());
    if (app.reasoner) config_vm->BindToReasoner(app.reasoner.get());
    // Register pipeline stage nodes for hot-reload support
    for (auto stage_id : {"capture", "preprocess", "inference", "detect",
                          "rule_eval", "reason", "export"}) {
        auto* node = app.pipeline->GetStage(stage_id);
        if (node) config_vm->RegisterStageNode(stage_id, node);
    }
    auto* dashboard_vm = new visualization::DashboardViewModel(&qapp);
    auto* frame_provider = new visualization::FrameProvider();

    // Live defect overlay: Pipe DetectionResult → DefectModel
    auto* defect_model_ptr = inspection_vm->defectModel();
    app.pipeline->SetDetectionCallback(
        [defect_model_ptr](const sai::detection::DetectionResult& dr) {
            QMetaObject::invokeMethod(
                defect_model_ptr, "UpdateDefects", Qt::QueuedConnection,
                Q_ARG(std::vector<sai::detection::RegionProposal>, dr.regions));
        });

    if (!app.evolution.has_value() && app.evolutions.empty()) {
        // No evolution — keep simple callback (no self-evolution capture needed)
        app.pipeline->SetResultCallback(
            [=](int fid, const reasoner::ReasoningResult& result) {
                inspection_vm->UpdateResult(fid, result);
                visualization::FrameSummary summary;
                summary.frame_id = fid;
                summary.verdict = result.verdict;
                summary.severity = std::to_string(result.severity);
                summary.timestamp = std::chrono::system_clock::now();
                dashboard_vm->AppendFrameSummary(std::move(summary));
            });
    } else {
        // Detection callback also needed when evolution is active
        app.pipeline->SetDetectionCallback(
            [defect_model_ptr](const sai::detection::DetectionResult& dr) {
                QMetaObject::invokeMethod(
                    defect_model_ptr, "UpdateDefects", Qt::QueuedConnection,
                    Q_ARG(std::vector<sai::detection::RegionProposal>, dr.regions));
            });
        app.pipeline->SetResultCallback(
            [&](int fid, const reasoner::ReasoningResult& result) {
                inspection_vm->UpdateResult(fid, result);
                visualization::FrameSummary summary;
                summary.frame_id = fid;
                summary.verdict = result.verdict;
                summary.severity = std::to_string(result.severity);
                summary.timestamp = std::chrono::system_clock::now();
                dashboard_vm->AppendFrameSummary(std::move(summary));

                // Single-position evolution
                if (app.evolution.has_value()) {
                    OfferToEvolution(*app.evolution, app.patch_core->LastContext(),
                                    result);
                }
                // Multi-position evolution
                if (!app.evolutions.empty()) {
                    AssembledApp::BankKey key{result.surface_id, result.position_id};
                    auto eit = app.evolutions.find(key);
                    if (eit != app.evolutions.end()) {
                        auto pit = app.patch_cores.find(key);
                        if (pit != app.patch_cores.end()) {
                            OfferToEvolution(eit->second, pit->second->LastContext(),
                                            result);
                        }
                    }
                }
            });
    }

    device::FakeCamera::Config cam_cfg{
        .width = 1024, .height = 1024, .fps = 10.0};
    auto camera = std::make_shared<device::FakeCamera>(cam_cfg);
    auto* pipeline_ptr = app.pipeline.get();
    auto* fp = frame_provider;
    auto frame_counter = std::make_shared<std::atomic<int>>(0);
    (void)camera->RegisterFrameCallback(
        [pipeline_ptr, fp, frame_counter](sai::image::RawImage img) {
            int fid = frame_counter->fetch_add(1, std::memory_order_relaxed);
            fp->RegisterRawFrame(fid, img);
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

    QObject::connect(&qapp, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
        (void)camera->StopAcquisition();
        (void)camera->Disconnect();
        (void)app.pipeline->Drain();
        if (app.evolution.has_value()) app.evolution->Stop();
        for (auto& [key, evo] : app.evolutions) {
            evo.Stop();
        }
        if (app.tuning_scheduler.has_value()) app.tuning_scheduler->Join();
        (void)app.pipeline->Stop();
        (void)app.ctx->Stop();
    });

    return qapp.exec();
}
