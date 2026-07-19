#include "gui_runner.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <nlohmann/json.hpp>

#include <sai/core/context.h>
#include <sai/pipeline/pipeline.h>
#include <sai/tuning/tuning_scheduler.h>
#include "app_builder.h"
#include "cli_args.h"
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

namespace {

/// Review mode: load review_index.json and populate ViewModels.
/// Does NOT start Pipeline / Camera / Tuning.
int RunReviewMode(const CliArgs& cli,
                  QGuiApplication& qapp,
                  QQmlApplicationEngine& engine,
                  sai::visualization::PipelineViewModel* pipeline_vm,
                  sai::visualization::InspectionViewModel* inspection_vm,
                  sai::visualization::DashboardViewModel* dashboard_vm,
                  sai::visualization::FrameProvider* frame_provider,
                  sai::visualization::ConfigViewModel* config_vm) {
    using json = nlohmann::json;

    auto index_path = std::filesystem::path(cli.review_dir) / "review_index.json";
    std::ifstream ifs(index_path);
    if (!ifs) {
        std::cerr << "Review mode: cannot open " << index_path << "\n";
        return 1;
    }

    json index;
    try {
        index = json::parse(ifs);
    } catch (const json::exception& e) {
        std::cerr << "Review mode: JSON parse error: " << e.what() << "\n";
        return 1;
    }

    std::string surface_id = index.value("surface_id", "unknown");
    int total_frames = index.value("total_frames", 0);
    std::cout << "Review mode: " << surface_id
              << " (" << total_frames << " frames)\n";

    // Populate DashboardViewModel with aggregate stats
    int ok = 0, ng = 0, warn = 0, uncertain = 0;
    for (auto& f : index["frames"]) {
        std::string v = f.value("verdict", "?");
        if (v == "OK") ++ok;
        else if (v == "NG") ++ng;
        else if (v == "WARN") ++warn;
        else if (v == "UNCERTAIN") ++uncertain;

        // Register frame path for lazy image loading
        int fid = f.value("frame_id", 0);
        std::string img_path = f.value("image_path", "");
        if (!img_path.empty()) {
            frame_provider->RegisterFramePath(
                fid, QString::fromStdString(img_path));
        }
    }

    // Seed DashboardViewModel with real frame IDs and verdicts from JSON.
    // Iterates the frames array directly so every verdict type (OK, NG, WARN,
    // UNCERTAIN) is included and each FrameSummary uses the real frame_id
    // from the file, matching what FrameProvider expects for lazy image loading.
    for (auto& f : index["frames"]) {
        sai::visualization::FrameSummary summary;
        summary.frame_id = f.value("frame_id", 0);
        summary.verdict = f.value("verdict", "?");
        if (summary.verdict == "OK") {
            summary.severity = "0.0";
        } else if (summary.verdict == "NG") {
            summary.severity = "1.0";
        } else if (summary.verdict == "WARN") {
            summary.severity = "0.5";
        } else if (summary.verdict == "UNCERTAIN") {
            summary.severity = "0.3";
        }
        summary.timestamp = std::chrono::system_clock::now();
        dashboard_vm->AppendFrameSummary(std::move(summary));
    }

    std::cout << "OK: " << ok << "  NG: " << ng
              << "  WARN: " << warn << "  UNCERTAIN: " << uncertain << "\n";

    // PipelineViewModel: no pipeline bound → shows "Stopped", which is correct
    // for review mode (no live pipeline running).

    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    engine.load("qrc:/MainWindow.qml");

    QObject::connect(&qapp, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
    });

    return qapp.exec();
}

}  // namespace

auto RunGui(int argc, char* argv[], AssembledApp& app, const CliArgs& cli) -> int {
    using namespace sai;

    QGuiApplication qapp(argc, argv);
    QQmlApplicationEngine engine;

    auto* pipeline_vm = new visualization::PipelineViewModel(&qapp);
    auto* inspection_vm = new visualization::InspectionViewModel(&qapp);
    auto* config_vm = new visualization::ConfigViewModel(&qapp);
    auto* dashboard_vm = new visualization::DashboardViewModel(&qapp);
    auto* frame_provider = new visualization::FrameProvider();

    // ── Review mode: bypass Pipeline / Camera / Tuning ──
    if (!cli.review_dir.empty()) {
        return RunReviewMode(cli, qapp, engine,
                             pipeline_vm, inspection_vm, dashboard_vm,
                             frame_provider, config_vm);
    }

    // ── Live mode: bind first pipeline to UI, create per-position cameras ──
    auto& primary_pipeline = app.GetPipeline();
    pipeline_vm->BindToPipeline(&primary_pipeline);
    config_vm->BindToPipeline(&primary_pipeline);
    config_vm->BindToRuleEngine(app.rule_engine.get());
    if (app.reasoner) config_vm->BindToReasoner(app.reasoner.get());
    for (auto stage_id : {"capture", "preprocess", "inference", "detect",
                          "rule_eval", "reason", "export"}) {
        auto* node = primary_pipeline.GetStage(stage_id);
        if (node) config_vm->RegisterStageNode(stage_id, node);
    }

    // Live defect overlay on the primary pipeline
    auto* defect_model_ptr = inspection_vm->defectModel();
    primary_pipeline.SetDetectionCallback(
        [defect_model_ptr](const sai::detection::DetectionResult& dr) {
            QMetaObject::invokeMethod(
                defect_model_ptr, "UpdateDefects", Qt::QueuedConnection,
                Q_ARG(std::vector<sai::detection::RegionProposal>, dr.regions));
        });

    // Result callback on primary pipeline (UI updates)
    bool has_evolution = false;
    for (auto& pp : app.positions) {
        if (pp.evolution != nullptr) { has_evolution = true; break; }
    }

    if (!has_evolution) {
        primary_pipeline.SetResultCallback(
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
        primary_pipeline.SetDetectionCallback(
            [defect_model_ptr](const sai::detection::DetectionResult& dr) {
                QMetaObject::invokeMethod(
                    defect_model_ptr, "UpdateDefects", Qt::QueuedConnection,
                    Q_ARG(std::vector<sai::detection::RegionProposal>, dr.regions));
            });
        primary_pipeline.SetResultCallback(
            [&](int fid, const reasoner::ReasoningResult& result) {
                inspection_vm->UpdateResult(fid, result);
                visualization::FrameSummary summary;
                summary.frame_id = fid;
                summary.verdict = result.verdict;
                summary.severity = std::to_string(result.severity);
                summary.timestamp = std::chrono::system_clock::now();
                dashboard_vm->AppendFrameSummary(std::move(summary));

                // Route to correct position's evolution
                auto* pp = app.FindPosition(result.surface_id, result.position_id);
                if (pp && pp->evolution != nullptr) {
                    seat_aoi::OfferToEvolution(*pp->evolution, pp->patch_core->LastContext(),
                                               result);
                }
            });
    }

    // Create one FakeCamera per position pipeline
    auto* fp = frame_provider;
    auto frame_counter = std::make_shared<std::atomic<int>>(0);
    std::vector<std::shared_ptr<device::FakeCamera>> cameras;
    for (auto& pp : app.positions) {
        device::FakeCamera::Config cam_cfg{
            .width = 1024, .height = 1024, .fps = 10.0};
        auto camera = std::make_shared<device::FakeCamera>(cam_cfg);
        auto* pl_ptr = pp.pipeline.get();
        (void)camera->RegisterFrameCallback(
            [pl_ptr, fp, frame_counter](sai::image::RawImage img) {
                int fid = frame_counter->fetch_add(1, std::memory_order_relaxed);
                fp->RegisterRawFrame(fid, img);
                (void)pl_ptr->Submit(std::move(img));
            });
        (void)camera->Connect();
        (void)camera->StartAcquisition();
        cameras.push_back(std::move(camera));
    }
    std::cout << "GUI: " << cameras.size() << " FakeCamera(s) → "
              << app.positions.size() << " pipeline(s)\n";

    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);
    pipeline_vm->StartRefresh(33);
    engine.load("qrc:/MainWindow.qml");

    QObject::connect(&qapp, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
        for (auto& cam : cameras) {
            (void)cam->StopAcquisition();
            (void)cam->Disconnect();
        }
        for (auto& pp : app.positions) {
            (void)pp.pipeline->Drain();
            if (pp.evolution != nullptr) pp.evolution->Stop();
            (void)pp.pipeline->Stop();
        }
        if (app.tuning_scheduler != nullptr) app.tuning_scheduler->Join();
        (void)app.ctx->Stop();
    });

    return qapp.exec();
}
