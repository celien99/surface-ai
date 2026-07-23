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

    auto resolve_resource = [](const std::string& path) {
        auto p = std::filesystem::path(path);
        return p.is_absolute() ? p : (ProjectRoot() / "apps/seat-aoi/resources" / p);
    };

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

    auto global_cfg = pipeline_yaml["pipeline"]["stages"][2]["config"]["global_model"];
    bool global_embedding_enabled =
        global_cfg.IsDefined() && global_cfg["enabled"].as<bool>(false);

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

    // =========================================================================
    // 4. FeatureBank + VectorPath — load coreset if provided
    // =========================================================================
    std::shared_ptr<detection::FeatureBank> feature_bank;
    std::shared_ptr<retrieval::VectorPath> vp;
    std::unique_ptr<detection::FeatureBank> detector_bank;

    if (!cli.coreset_path.empty()) {
        pc_cfg.feature_bank_path = cli.coreset_path;

        auto fb_result = detection::FeatureBank::LoadFromFile(cli.coreset_path, kEmbedDim);
        if (fb_result) {
            auto loaded_detector_bank = std::make_unique<detection::FeatureBank>(
                std::move(*fb_result));
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
            auto detector_gpu_result = loaded_detector_bank->ToGpu();
            if (!detector_gpu_result) {
                return tl::make_unexpected(detector_gpu_result.error());
            }
#endif
            detector_bank = std::move(loaded_detector_bank);
            auto retrieval_bank = detection::FeatureBank::LoadFromFile(
                cli.coreset_path, kEmbedDim);
            if (!retrieval_bank) return tl::make_unexpected(retrieval_bank.error());
            feature_bank = std::make_shared<detection::FeatureBank>(
                std::move(*retrieval_bank));
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
            auto retrieval_gpu_result = feature_bank->ToGpu();
            if (!retrieval_gpu_result) {
                return tl::make_unexpected(retrieval_gpu_result.error());
            }
#endif
            vp = std::make_shared<retrieval::VectorPath>(*feature_bank);
        } else {
            return tl::make_unexpected(fb_result.error());
        }
    }

    if (!detector_bank) {
        std::cout << "FeatureBank: not loaded (detection will use PCA-only mode)\n";
    }

    auto patch_core = std::make_shared<detection::PatchCore>(pc_cfg);
    if (detector_bank) {
        patch_core->SetFeatureBank(std::move(detector_bank));
        auto bank = patch_core->GetFeatureBankSnapshot();
        std::cout << "FeatureBank loaded: " << bank->NumSamples()
                  << " samples, dim=" << bank->Dim() << "\n";
        std::cout << "VectorPath: ready\n";
    }
    auto patch_core_init = patch_core->Initialize(*ctx);
    if (!patch_core_init) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "PatchCore initialization failed: " + patch_core_init.error().message});
    }

    // =========================================================================
    // 4b. Multi-position coreset loading (--coreset-manifest)
    // =========================================================================
    std::map<sai::pipeline::BankKey, std::shared_ptr<detection::PatchCore>> patch_cores;
    std::map<sai::pipeline::BankKey, std::filesystem::path> patch_core_paths;
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
                auto position_bank = std::make_unique<detection::FeatureBank>(
                    std::move(*fb_result));
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
                auto position_gpu_result = position_bank->ToGpu();
                if (!position_gpu_result) {
                    return tl::make_unexpected(position_gpu_result.error());
                }
#endif
                pos_patch_core->SetFeatureBank(std::move(position_bank));
                auto init_result = pos_patch_core->Initialize(*ctx);
                if (!init_result) {
                    return tl::make_unexpected(ErrorInfo{
                        ErrorCode::Detection_FeatureBankLoadFailed,
                        "PatchCore initialization failed for position " +
                            std::to_string(bank.position_id) + ": " +
                            init_result.error().message});
                }
                patch_cores[key] = pos_patch_core;
                patch_core_paths[key] = bank.path;

                std::cout << "  Position " << bank.position_id << ": "
                          << pos_pc_cfg.k_nearest << " k-NN ready\n";
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
                        auto profile_path = patch_core_paths[key];
                        profile_path = std::filesystem::path(
                            profile_path.string() + ".profile.yaml");
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
                    auto profile_path = std::filesystem::path(
                        cli.coreset_path + std::string(".profile.yaml"));
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
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Reasoner_TreeLoadFailed,
            "Decision tree load failed: " + tree_result.error().message});
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
        auto image_pool_arena = std::make_unique<memory::ArenaAllocator>(4096);
        auto image_pool_result = memory::GpuPool::Create(
            memory::MemoryPoolConfig{
                .slab_size = kImageSize * kImageSize * 3,
                .slab_count = 1,
            },
            *image_pool_arena);
        if (!image_pool_result) return tl::make_unexpected(image_pool_result.error());
        auto image_pool = std::shared_ptr<memory::GpuPool>(
            std::move(*image_pool_result));

        auto embedding_pool_arena = std::make_unique<memory::ArenaAllocator>(4096);
        auto embedding_pool_result = memory::GpuPool::Create(
            memory::MemoryPoolConfig{
                .slab_size = kEmbedDim * (kImageSize / kPatchSize)
                    * (kImageSize / kPatchSize) * sizeof(float),
                .slab_count = 4,
            },
            *embedding_pool_arena);
        if (!embedding_pool_result) {
            return tl::make_unexpected(embedding_pool_result.error());
        }
        auto embedding_pool = std::shared_ptr<memory::GpuPool>(
            std::move(*embedding_pool_result));

        auto embedder_result = seat_aoi::CreateDinoV2PatchEmbedder(
            DinoV2Engine(), kImageSize, kPatchSize, kEmbedDim);
        if (!embedder_result) {
            return tl::make_unexpected(embedder_result.error());
        }
        auto position_embedder = std::move(*embedder_result);
        position_embedder->SetGpuPool(embedding_pool.get());

        std::shared_ptr<embedding::IEmbedder> position_global_embedder;
        if (global_embedding_enabled) {
            auto global_result = seat_aoi::CreateClipGlobalEmbedder(
                ClipEngine(), 224, 512);
            if (!global_result) return tl::make_unexpected(global_result.error());
            position_global_embedder = std::move(*global_result);
        }

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

        // Wire per-position inference resources and shared stateless components.
        if (auto* s = pos_pipeline->GetStage("inference")) {
            static_cast<pipeline::InferenceStage*>(s)->SetEmbedder(position_embedder);
            static_cast<pipeline::InferenceStage*>(s)->SetGpuPool(image_pool.get());
            if (position_global_embedder)
                static_cast<pipeline::InferenceStage*>(s)->SetGlobalEmbedder(
                    position_global_embedder);
        }
        if (auto* s = pos_pipeline->GetStage("rule_eval")) {
            auto* rs = static_cast<pipeline::RuleEvalStage*>(s);
            rs->SetRuleFile(resolve_resource("rules/seat_leather_defects.yaml"));
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
            pos_def.pc->SetContextCaptureEnabled(true);
            auto* evo_raw = pos_def.evo.get();
            pos_pipeline->SetResultCallback(
                [evo_raw](const std::shared_ptr<const pipeline::FrameContext>& frame,
                          const reasoner::ReasoningResult& result) {
                    if (frame && frame->patchcore_context) {
                        seat_aoi::OfferToEvolution(
                            *evo_raw, *frame->patchcore_context, result);
                    }
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
        pp.image_pool_arena = std::move(image_pool_arena);
        pp.embedding_pool_arena = std::move(embedding_pool_arena);
        pp.image_pool = std::move(image_pool);
        pp.embedding_pool = std::move(embedding_pool);
        pp.embedder = std::move(position_embedder);
        pp.global_embedder = std::move(position_global_embedder);
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
                tuning_scheduler->Start(tuning_stop_source.get_token());
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
    app.rule_engine = rule_engine;
    app.exporter = exporter;
    app.knowledge_store = std::move(ks);
    app.kg = kg;
    app.kg_evolution = kg_evolution;
    app.sam2_segmenter = sam2_segmenter;
    app.reasoner = reasoner;
    app.feature_bank = feature_bank;
    app.vector_path = vp;
    app.tuning_scheduler = std::move(tuning_scheduler);
    app.tuning_stop_source = std::move(tuning_stop_source);
    app.positions = std::move(position_pipelines);

    return app;
}
