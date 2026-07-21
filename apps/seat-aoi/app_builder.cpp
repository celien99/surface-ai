#include "app_builder.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stop_token>
#include <string>

#include <yaml-cpp/yaml.h>

#include "app_config.h"
#include "cli_args.h"
#include "embedder_factory.h"
#include "evolution_offer.h"
#include "knowledge_seed.h"
#include "tuning_wiring.h"

#include "pipeline/stage_nodes.h"

#include <sai/core/context.h>
#include <sai/core/error.h>
#include <sai/inference/tensorrt_engine.h>
#include <sai/inference/clip_adapter.h>
#include <sai/inference/sam2_segmenter.h>
#include <sai/embedding/embedder.h>
#include <sai/memory/arena_allocator.h>
#include <sai/memory/gpu_pool.h>
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
#include <sai/io/coreset_manifest.h>
#include <sai/io/exporter.h>
#include <sai/pipeline/pipeline.h>
#include <sai/pipeline/pipeline_config.h>

#include <cuda_runtime.h>

// Out-of-line destructor — all sai:: types are complete at this point
AssembledApp::~AssembledApp() = default;
PositionPipeline::~PositionPipeline() = default;

auto AssembledApp::HasPosition(const std::string& sid, std::uint16_t pid) const -> bool {
    for (auto& pp : positions)
        if (pp.key.first == sid && pp.key.second == pid) return true;
    return false;
}
auto AssembledApp::FindPosition(const std::string& sid, std::uint16_t pid) -> PositionPipeline* {
    for (auto& pp : positions)
        if (pp.key.first == sid && pp.key.second == pid) return &pp;
    return nullptr;
}

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
        pipeline_yaml = YAML::LoadFile(PipelineYaml().string());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "Failed to load pipeline.yaml: " + std::string(e.what())});
    }

    // =========================================================================
    // 2. Inference / embedding setup — TensorRT + DINOv2 (768-dim)
    // =========================================================================

    auto embedder_result = seat_aoi::CreateDinoV2PatchEmbedder(
        DinoV2Engine(), kImageSize, kPatchSize, kEmbedDim);
    if (!embedder_result) return tl::make_unexpected(std::move(embedder_result.error()));
    auto embedder = std::move(*embedder_result);

    // Wire GpuPool for zero-copy GPU feature extraction (D2D instead of D2H).
    std::shared_ptr<sai::memory::GpuPool> gpu_pool_owner;
    {
        static sai::memory::ArenaAllocator gpu_pool_arena(4096);
        sai::memory::MemoryPoolConfig pool_cfg;
        pool_cfg.slab_size = kEmbedDim * (kImageSize / kPatchSize) *
                              (kImageSize / kPatchSize) * sizeof(float);
        pool_cfg.slab_count = 4;
        auto gpu_pool = sai::memory::GpuPool::Create(pool_cfg, gpu_pool_arena);
        if (gpu_pool) {
            embedder->SetGpuPool(gpu_pool->get());
            gpu_pool_owner = std::shared_ptr<sai::memory::GpuPool>(
                std::move(*gpu_pool));
            std::cout << "GpuPool: " << pool_cfg.slab_count << " slabs × "
                      << (pool_cfg.slab_size >> 20) << " MiB"
                      << " (zero-copy embedding)\n";
        } else {
            std::cerr << "Warning: GpuPool creation failed — "
                      << "embedding will use CPU fallback\n";
        }
    }

    // Global embedder (CLIP) for cross-modal vector retrieval.
    std::shared_ptr<embedding::IEmbedder> global_embedder;
    {
        auto global_cfg = pipeline_yaml["pipeline"]["stages"][2]["config"]["global_model"];
        if (global_cfg.IsDefined() && global_cfg["enabled"].as<bool>(false)) {
            if (!std::filesystem::exists(ClipEngine())) {
                std::cerr << "Warning: CLIP engine not found at "
                          << ClipEngine() << " — skipping global embedder\n";
            } else {
                auto clip_engine = std::make_shared<inference::TensorRtEngine>(
                    /*device_ordinal=*/0);
                inference::ClipConfig clip_cfg;
                clip_cfg.engine_path = ClipEngine();
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
        pc_cfg.feature_bank_path = cli.coreset_path;

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
    // 4b. Multi-position coreset loading (--coreset-manifest)
    // =========================================================================
    std::map<sai::pipeline::BankKey, std::shared_ptr<detection::PatchCore>> patch_cores;
    std::map<sai::pipeline::BankKey, std::unique_ptr<detection::CoresetEvolution>> evolutions_map;
    std::map<sai::pipeline::BankKey, std::stop_source> evo_stop_sources_map;

    if (!cli.coreset_manifest_path.empty()) {
        auto manifest_result = io::LoadCoresetManifest(cli.coreset_manifest_path);
        if (!manifest_result) {
            std::cerr << "Warning: failed to load coreset manifest: "
                      << manifest_result.error().message << "\n";
        } else {
            auto& manifest = *manifest_result;
            std::cout << "Loading coreset manifest: " << manifest.surface_id
                      << " (" << manifest.banks.size() << " positions)\n";

            for (auto& bank : manifest.banks) {
                sai::pipeline::BankKey key{manifest.surface_id, bank.position_id};

                auto fb_result = detection::FeatureBank::LoadFromFile(bank.path, kEmbedDim);
                if (!fb_result) {
                    std::cerr << "Warning: failed to load coreset for position "
                              << bank.position_id << ": " << fb_result.error().message << "\n";
                    continue;
                }

                auto pos_pc_cfg = pc_cfg;
                pos_pc_cfg.feature_bank_path = bank.path;
                auto pos_patch_core = std::make_shared<detection::PatchCore>(pos_pc_cfg);
                pos_patch_core->SetFeatureBank(
                    std::make_unique<detection::FeatureBank>(std::move(*fb_result)));
                patch_cores[key] = pos_patch_core;

                std::cout << "  Position " << bank.position_id << ": "
                          << pos_patch_core->LastContext().k_nearest << " k-NN ready\n";
            }
        }
    }

    // =========================================================================
    // 4c. Coreset Self-Evolution — YAML config-driven
    // =========================================================================
    std::unique_ptr<detection::CoresetEvolution> evolution;
    {
        auto se_node = pipeline_yaml["pipeline"]["stages"][3]["config"]["self_evolution"];
        if (se_node.IsDefined()) {
            auto evo_cfg_result = detection::EvolutionConfig::FromYaml(se_node);
            if (evo_cfg_result.has_value() && evo_cfg_result->enabled) {
                auto evo_cfg = std::move(*evo_cfg_result);

                if (!patch_cores.empty()) {
                    for (auto& [key, pc] : patch_cores) {
                        auto profile_path = std::filesystem::path(
                            pc_cfg.feature_bank_path).replace_extension(".profile.yaml");
                        std::optional<detection::NormalityProfile> profile;
                        if (std::filesystem::exists(profile_path)) {
                            auto r = detection::NormalityProfile::LoadFromYaml(profile_path);
                            if (r.has_value()) profile = std::move(*r);
                        }
                        if (profile.has_value()) {
                            auto ts = evo_cfg.target_size;
                            auto evo = std::make_unique<detection::CoresetEvolution>(
                                evo_cfg, *pc, std::move(*profile));
                            evolutions_map.try_emplace(key, std::move(evo));
                            std::cout << "CoresetEvolution: pos " << key.second
                                      << " enabled (target_size=" << ts << ")\n";
                        } else {
                            std::cout << "CoresetEvolution: pos " << key.second
                                      << " skipped (no profile)\n";
                        }
                    }
                } else if (feature_bank) {
                    auto profile_path = std::filesystem::path(cli.coreset_path)
                        .replace_extension(".profile.yaml");
                    std::optional<detection::NormalityProfile> profile;
                    if (std::filesystem::exists(profile_path)) {
                        auto r = detection::NormalityProfile::LoadFromYaml(profile_path);
                        if (r.has_value()) profile = std::move(*r);
                    }
                    if (!profile.has_value())
                        profile = detection::NormalityProfile::Compute(*feature_bank);
                    if (profile.has_value()) {
                        auto ts = evo_cfg.target_size;
                        evolution = std::make_unique<detection::CoresetEvolution>(
                            std::move(evo_cfg), *patch_core, std::move(*profile));
                        std::cout << "CoresetEvolution: enabled (target_size=" << ts << ")\n";
                    }
                }
            }
        }
    }

    // =========================================================================
    // 5. KnowledgeStore — unified KG + Evolution facade (shared across positions)
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

    auto kg = std::shared_ptr<knowledge::KnowledgeGraph>(&ks->Graph(), [](auto*) {});
    auto kg_evolution = std::shared_ptr<knowledge::KnowledgeEvolution>(
        &ks->Evolution(), [](auto*) {});

    // =========================================================================
    // 5b. SAM2 segmenter (M5 placeholder — shared across positions).
    // =========================================================================
    std::shared_ptr<inference::Sam2Segmenter> sam2_segmenter;
    {
        auto sam2_cfg = pipeline_yaml["pipeline"]["stages"][5]["config"]["sam2"];
        if (sam2_cfg.IsDefined() && sam2_cfg["enabled"].as<bool>(false)) {
            if (!std::filesystem::exists(Sam2Engine())) {
                std::cerr << "Warning: SAM2 engine not found at "
                          << Sam2Engine() << " — skipping segmenter\n";
            } else {
                auto sam2_engine = std::make_shared<inference::TensorRtEngine>(
                    /*device_ordinal=*/0);
                inference::Sam2Config s2_cfg;
                s2_cfg.engine_path = Sam2Engine();
                s2_cfg.image_size = kImageSize;

                auto sam2_adapter = inference::Sam2Adapter::Create(*sam2_engine, s2_cfg);
                if (sam2_adapter) {
                    auto seg_result = inference::Sam2Segmenter::Create(
                        std::move(*sam2_adapter));
                    if (seg_result) {
                        sam2_segmenter = std::make_shared<inference::Sam2Segmenter>(
                            std::move(*seg_result));
                        std::cout << "Sam2Segmenter: enabled\n";
                    }
                } else {
                    std::cerr << "Warning: Sam2Adapter creation failed: "
                              << sam2_adapter.error().message << "\n";
                }
            }
        }
    }

    // =========================================================================
    // 6. Rule Engine + Reasoner — shared across all positions
    // =========================================================================
    auto rule_engine = std::make_shared<rule::RuleEngine>();

    auto tree_result = reasoner::DecisionTree::LoadFromYAML(DecisionTree().string());
    std::shared_ptr<reasoner::IReasoner> reasoner;
    if (tree_result) {
        reasoner = std::make_shared<reasoner::DefaultReasoner>(std::move(*tree_result));
    } else {
        std::cerr << "Warning: decision tree load failed, reasoner unavailable\n";
    }

    // =========================================================================
    // 7. Export — shared across all positions
    // =========================================================================
    auto exporter = std::make_shared<io::JsonExporter>();

    // =========================================================================
    // 8. Build per-position pipeline list
    // =========================================================================
    struct PositionDef {
        sai::pipeline::BankKey key;
        std::shared_ptr<detection::PatchCore> pc;
        std::unique_ptr<detection::CoresetEvolution> evo;
    };
    std::vector<PositionDef> position_defs;

    if (!patch_cores.empty()) {
        for (auto& [key, pc] : patch_cores) {
            PositionDef def;
            def.key = key;
            def.pc = pc;
            auto eit = evolutions_map.find(key);
            if (eit != evolutions_map.end()) def.evo = std::move(eit->second);
            position_defs.push_back(std::move(def));
        }
    } else {
        PositionDef def;
        def.key = sai::pipeline::BankKey{"default", 0};
        def.pc = patch_core;
        if (evolution) def.evo = std::move(evolution);
        position_defs.push_back(std::move(def));
    }

    // =========================================================================
    // 9. Create one Pipeline per position — each runs independently
    // =========================================================================
    std::vector<PositionPipeline> position_pipelines;

    for (auto& pos_def : position_defs) {
        auto pipeline_result = pipeline::Pipeline::LoadFromYAML(
            PipelineYaml().string(), *ctx);
        if (!pipeline_result) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Pipeline_InvalidConfig,
                "Pipeline load failed for position " + std::to_string(pos_def.key.second)
                + ": " + pipeline_result.error().message});
        }
        auto pos_pipeline = std::move(*pipeline_result);

        // Wire per-position detector
        if (auto* s = pos_pipeline->GetStage("detect")) {
            auto* ds = static_cast<pipeline::DetectStage*>(s);
            ds->AddDetector(pos_def.key.first, pos_def.key.second, pos_def.pc);
        }

        // Wire shared components into this pipeline's stages
        if (auto* s = pos_pipeline->GetStage("inference")) {
            static_cast<pipeline::InferenceStage*>(s)->SetEmbedder(embedder);
            if (gpu_pool_owner)
                static_cast<pipeline::InferenceStage*>(s)->SetGpuPool(gpu_pool_owner.get());
            if (global_embedder)
                static_cast<pipeline::InferenceStage*>(s)->SetGlobalEmbedder(global_embedder);
        }
        if (auto* s = pos_pipeline->GetStage("rule_eval")) {
            auto* rs = static_cast<pipeline::RuleEvalStage*>(s);
            rs->SetRuleEngine(rule_engine);
            rs->SetKnowledgeGraph(kg);
            if (vp) rs->SetVectorPath(vp);
        }
        if (auto* s = pos_pipeline->GetStage("reason")) {
            static_cast<pipeline::ReasonStage*>(s)->SetReasoner(reasoner);
            if (sam2_segmenter)
                static_cast<pipeline::ReasonStage*>(s)->SetSam2Segmenter(sam2_segmenter);
        }
        if (auto* s = pos_pipeline->GetStage("export")) {
            static_cast<pipeline::ExportStage*>(s)->SetExporter(exporter);
            if (!cli.output_dir.empty())
                static_cast<pipeline::ExportStage*>(s)->SetOutputDir(cli.output_dir);
        }

        // Per-position result callback → evolution routing
        if (pos_def.evo != nullptr) {
            auto* pc_raw = pos_def.pc.get();
            auto* evo_raw = pos_def.evo.get();
            pos_pipeline->SetResultCallback(
                [pc_raw, evo_raw](int /*fid*/, const reasoner::ReasoningResult& result) {
                    seat_aoi::OfferToEvolution(*evo_raw, pc_raw->LastContext(), result);
                });
        }

        // Start pipeline
        auto start_result = pos_pipeline->Start();
        if (!start_result) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Pipeline_InvalidConfig,
                "Pipeline start failed for position " + std::to_string(pos_def.key.second)
                + ": " + start_result.error().message});
        }

        // Start evolution
        PositionPipeline pp;
        pp.key = pos_def.key;
        pp.pipeline = std::move(pos_pipeline);
        pp.patch_core = pos_def.pc;
        if (pos_def.evo != nullptr) {
            pp.evolution_stop_source = std::stop_source{};
            pos_def.evo->Start(pp.evolution_stop_source.get_token());
        }
        pp.evolution = std::move(pos_def.evo);
        position_pipelines.push_back(std::move(pp));

        std::cout << "Pipeline: position " << pos_def.key.second
                  << " (" << pos_def.key.first << ") ready\n";
    }

    // =========================================================================
    // 10. Bayesian Auto-Tuning (optional, shared across all positions)
    // =========================================================================
    std::unique_ptr<tuning::TuningScheduler> tuning_scheduler;
    std::stop_source tuning_stop_source;
    {
        auto tuning_node = pipeline_yaml["pipeline"]["tuning"];
        if (tuning_node.IsDefined() && tuning_node["enabled"].as<bool>(false) && ks) {
            auto tuning_result = TryCreateTuningScheduler(
                pipeline_yaml, *kg, *kg_evolution, *reasoner,
                *position_pipelines[0].pipeline);
            if (tuning_result.has_value() && *tuning_result) {
                tuning_scheduler = std::move(*tuning_result);
                std::cout << "TuningScheduler: started (GP+EI)\n";
            } else {
                std::cerr << "Warning: failed to start tuning scheduler: "
                          << tuning_result.error().message << "\n";
            }
        }
    }

    // =========================================================================
    // 11. Assemble and return
    // =========================================================================
    AssembledApp app;
    app.ctx = std::move(ctx);
    app.embedder = embedder;
    app.gpu_pool = std::move(gpu_pool_owner);
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
    app.tuning_scheduler = std::move(tuning_scheduler);
    app.tuning_stop_source = std::move(tuning_stop_source);
    app.positions = std::move(position_pipelines);

    return app;
}
