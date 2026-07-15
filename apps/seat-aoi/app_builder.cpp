#include "app_builder.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>

#include <yaml-cpp/yaml.h>

#include "app_config.h"
#include "cli_args.h"
#include "knowledge_seed.h"
#include "tuning_wiring.h"

#include "pipeline/stage_nodes.h"

#include <sai/core/context.h>
#include <sai/core/error.h>
#include <sai/inference/tensorrt_engine.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/inference/clip_adapter.h>
#include <sai/inference/sam2_segmenter.h>
#include <sai/embedding/patch_embedder.h>
#include <sai/embedding/global_embedder.h>
#include <sai/embedding/embedder.h>
#include <sai/detection/patch_core.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/coreset_evolution.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_store.h>
#include <sai/knowledge/knowledge_evolution.h>
#include <sai/retrieval/vector_path.h>
#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>
#include <sai/reasoner/decision_tree.h>
#include <sai/io/exporter.h>
#include <sai/pipeline/pipeline.h>
#include <sai/pipeline/pipeline_config.h>

#include <cuda_runtime.h>

auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp> {
    using namespace sai;
    using namespace seat_aoi::config;

    // =========================================================================
    // 1. DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // Load pipeline.yaml once — all sub-steps share this parse result
    // =========================================================================
    YAML::Node pipeline_yaml;
    try {
        pipeline_yaml = YAML::LoadFile(kPipelineYaml.data());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "Failed to load pipeline.yaml: " + std::string(e.what())});
    }

    // =========================================================================
    // 2. Inference / embedding setup — TensorRT + DINOv3 (1024-dim)
    // =========================================================================
    auto infer_engine = std::make_shared<inference::TensorRtEngine>(/*device_ordinal=*/0);
    std::cout << "Inference: TensorRtEngine (GPU)\n";

    inference::DinoV3Config dino_cfg;
    dino_cfg.engine_path = kDinoV3Engine;
    dino_cfg.image_size = kImageSize;
    dino_cfg.patch_size = kPatchSize;
    dino_cfg.embed_dim = kEmbedDim;

    auto dino_adapter = inference::DinoV3Adapter::Create(*infer_engine, dino_cfg);
    if (!dino_adapter) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "DinoV3Adapter creation failed: " + dino_adapter.error().message});
    }
    auto patch_emb = embedding::PatchEmbedder::Create(std::move(*dino_adapter));
    if (!patch_emb) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder creation failed"});
    }
    auto embedder = std::make_shared<embedding::PatchEmbedder>(std::move(*patch_emb));
    std::cout << "Embedder: PatchEmbedder (DINOv3)\n";

    // Global embedder (CLIP) for cross-modal vector retrieval.
    // Enabled via pipeline.yaml → stages[2].config.global_model.enabled
    std::shared_ptr<embedding::IEmbedder> global_embedder;
    {
        auto global_cfg = pipeline_yaml["pipeline"]["stages"][2]["config"]["global_model"];
        if (global_cfg.IsDefined() && global_cfg["enabled"].as<bool>(false)) {
            auto clip_engine = std::make_shared<inference::TensorRtEngine>(
                /*device_ordinal=*/0);
            inference::ClipConfig clip_cfg;
            clip_cfg.engine_path = kClipEngine;
            clip_cfg.image_size = 224;
            clip_cfg.embed_dim = 512;

            auto clip_adapter = inference::ClipAdapter::Create(
                *clip_engine, clip_cfg);
            if (clip_adapter) {
                auto global_emb = embedding::GlobalEmbedder::Create(
                    std::move(*clip_adapter));
                if (global_emb) {
                    global_embedder = std::make_shared<embedding::GlobalEmbedder>(
                        std::move(*global_emb));
                    std::cout << "GlobalEmbedder: CLIP (enabled)\n";
                } else {
                    std::cerr << "Warning: GlobalEmbedder creation failed: "
                              << global_emb.error().message << "\n";
                }
            } else {
                std::cerr << "Warning: ClipAdapter creation failed: "
                          << clip_adapter.error().message << "\n";
            }
        }
    }

    // =========================================================================
    // 3. Detection — PatchCore with matched embed_dim
    // =========================================================================
    detection::PatchCore::Config pc_cfg;
    pc_cfg.embed_dim = kEmbedDim;
    pc_cfg.image_width = kImageSize;
    pc_cfg.image_height = kImageSize;
    pc_cfg.patch_size = kPatchSize;
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
    // 4b. Coreset Self-Evolution — YAML config-driven
    // =========================================================================
    std::optional<detection::CoresetEvolution> evolution;
    if (feature_bank) {
        if (auto se_node = pipeline_yaml["pipeline"]["stages"][3]["config"]["self_evolution"];
            se_node.IsDefined()) {
            auto evo_cfg = detection::EvolutionConfig::FromYaml(se_node);
            if (evo_cfg.has_value() && evo_cfg->enabled) {
                auto profile_path = std::filesystem::path(cli.coreset_path)
                    .replace_extension(".profile.yaml");
                auto profile = std::filesystem::exists(profile_path)
                    ? detection::NormalityProfile::LoadFromYaml(profile_path)
                    : detection::NormalityProfile::Compute(*feature_bank);

                if (profile.has_value()) {
                    evolution.emplace(
                        std::move(*evo_cfg), *patch_core, std::move(*profile));
                    std::cout << "CoresetEvolution: enabled (target_size="
                              << evo_cfg->target_size << ")\n";
                }
            }
        }
    }

    // =========================================================================
    // 5. KnowledgeStore — unified KG + Evolution facade
    // =========================================================================
    auto ks_result = knowledge::KnowledgeStore::Create(
        knowledge::KnowledgeStore::Config{":memory:", true});
    if (!ks_result) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "KnowledgeStore creation failed: " + ks_result.error().message});
    }
    auto ks = std::move(*ks_result);
    SeedKnowledgeGraph(ks->Graph());

    // Non-owning shared_ptr wrappers — the store owns the objects
    auto kg = std::shared_ptr<knowledge::KnowledgeGraph>(
        &ks->Graph(), [](auto*) {});
    auto kg_evolution = std::shared_ptr<knowledge::KnowledgeEvolution>(
        &ks->Evolution(), [](auto*) {});

    // =========================================================================
    // 5b. SAM2 segmenter (M5 placeholder — wired but not yet activated).
    // =========================================================================
    std::shared_ptr<inference::Sam2Segmenter> sam2_segmenter;
    {
        auto sam2_cfg = pipeline_yaml["pipeline"]["stages"][5]["config"]["sam2"];
        if (sam2_cfg.IsDefined() && sam2_cfg["enabled"].as<bool>(false)) {
            auto sam2_engine = std::make_shared<inference::TensorRtEngine>(
                /*device_ordinal=*/0);
            inference::Sam2Config s2_cfg;
            s2_cfg.engine_path = kSam2Engine;
            s2_cfg.image_size = kImageSize;

            auto sam2_adapter = inference::Sam2Adapter::Create(
                *sam2_engine, s2_cfg);
            if (sam2_adapter) {
                auto seg_result = inference::Sam2Segmenter::Create(
                    std::move(*sam2_adapter));
                if (seg_result) {
                    sam2_segmenter =
                        std::make_shared<inference::Sam2Segmenter>(
                            std::move(*seg_result));
                    std::cout << "Sam2Segmenter: enabled\n";
                }
            } else {
                std::cerr << "Warning: Sam2Adapter creation failed: "
                          << sam2_adapter.error().message << "\n";
            }
        }
    }

    // =========================================================================
    // 6. Rule Engine + Reasoner — loaded from YAML
    // =========================================================================
    auto rule_engine = std::make_shared<rule::RuleEngine>();

    auto tree_result = reasoner::DecisionTree::LoadFromYAML(
        std::string(kDecisionTree));
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
        std::string(kPipelineYaml), *ctx);
    if (!pipeline_result) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "Pipeline load failed: " + pipeline_result.error().message});
    }
    auto pipeline = std::move(*pipeline_result);

    // =========================================================================
    // 9. Wire all business objects into Pipeline stages
    // =========================================================================
    if (auto* s = pipeline->GetStage("inference")) {
        static_cast<pipeline::InferenceStage*>(s)->SetEmbedder(embedder);
        if (global_embedder)
            static_cast<pipeline::InferenceStage*>(s)->SetGlobalEmbedder(
                global_embedder);
    }
    if (auto* s = pipeline->GetStage("detect"))
        static_cast<pipeline::DetectStage*>(s)->SetDetector(patch_core);
    if (auto* s = pipeline->GetStage("rule_eval")) {
        auto* rs = static_cast<pipeline::RuleEvalStage*>(s);
        rs->SetRuleEngine(rule_engine);
        rs->SetKnowledgeGraph(kg);
        if (vp) rs->SetVectorPath(vp);
    }
    if (auto* s = pipeline->GetStage("reason")) {
        static_cast<pipeline::ReasonStage*>(s)->SetReasoner(reasoner);
        if (sam2_segmenter)
            static_cast<pipeline::ReasonStage*>(s)->SetSam2Segmenter(
                sam2_segmenter);
    }
    if (auto* s = pipeline->GetStage("export"))
        static_cast<pipeline::ExportStage*>(s)->SetExporter(exporter);

    // =========================================================================
    // 9c. Bayesian Auto-Tuning (optional, guarded by pipeline.yaml tuning section)
    // =========================================================================
    std::optional<tuning::TuningScheduler> tuning_scheduler;
    std::stop_source tuning_stop_source;

    if (reasoner) {
        auto tuning_result = TryCreateTuningScheduler(
            pipeline_yaml, ks->Graph(), ks->Evolution(), *reasoner, *pipeline);
        if (!tuning_result) {
            std::cerr << "Tuning: setup failed: "
                      << tuning_result.error().message << "\n";
        } else if (tuning_result->has_value()) {
            tuning_scheduler.emplace(std::move(**tuning_result));
            // 6. Start tuning thread
            tuning_scheduler->Start(tuning_stop_source.get_token());
        }
    }

    // =========================================================================
    // 9b. Result callback for self-evolution (shared by headless and GUI modes)
    //    Note: The callback set here is the evolution-aware variant.
    //    GuiRunner will override it if evolution is disabled.
    // =========================================================================
    if (evolution.has_value()) {
        pipeline->SetResultCallback(
            [patch_core, &evo = *evolution]
            (int fid, const reasoner::ReasoningResult& result) {
                (void)fid;
                if (evo.IsRunning()) {
                    const auto& ctx = patch_core->LastContext();
                    if (!ctx.knn_distances.empty()) {
                        evo.AssessAndOffer(
                            ctx.knn_distances.data(),
                            ctx.knn_distances.size() / ctx.k_nearest,
                            ctx.k_nearest,
                            ctx.embedding_data.data(),
                            ctx.grid_h,
                            ctx.grid_w,
                            ctx.dim,
                            ctx.detection_result,
                            result.triggered_rules.size(),  // matched_rules_count
                            result.verdict,                 // reasoner_verdict
                            ctx.effective_threshold,
                            ctx.pca_image_score,
                            ctx.pca_self_query_p95);
                    }
                }
            });
    }

    // =========================================================================
    // 10. Start Pipeline
    // =========================================================================
    auto start_result = pipeline->Start();
    if (!start_result) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "Pipeline start failed: " + start_result.error().message});
    }

    // =========================================================================
    // 10b. Start self-evolution background thread
    // =========================================================================
    std::stop_source evo_ss;
    if (evolution.has_value()) {
        evolution->Start(evo_ss.get_token());
        std::cout << "CoresetEvolution: started\n";
    }

    // =========================================================================
    // 11. Assemble and return
    // =========================================================================
    AssembledApp app;
    app.ctx = std::move(ctx);
    app.pipeline = std::move(pipeline);
    app.embedder = embedder;
    app.patch_core = patch_core;
    app.rule_engine = rule_engine;
    app.exporter = exporter;
    app.knowledge_store = std::move(ks);
    app.kg = kg;
    app.kg_evolution = kg_evolution;
    app.global_embedder = global_embedder;
    app.sam2_segmenter = sam2_segmenter;
    app.reasoner = reasoner;
    app.feature_bank = feature_bank;
    app.vector_path = vp;
    app.evolution = std::move(evolution);
    app.tuning_scheduler = std::move(tuning_scheduler);
    app.tuning_stop_source = std::move(tuning_stop_source);
    app.evolution_stop_source = std::move(evo_ss);

    return app;
}
