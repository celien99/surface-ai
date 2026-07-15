#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>
#include <sstream>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <sqlite3.h>

#include "sai/core/context.h"
#include "pipeline/stage_nodes.h"
#include "sai/device/fake_camera.h"
#include "sai/image/raw_image.h"
#include "sai/image/surface_image.h"
#include "sai/image/preprocess.h"
#include "sai/io/importer.h"
#include "sai/io/exporter.h"

// Inference: TensorRT
#include <sai/inference/tensorrt_engine.h>
#include <sai/inference/clip_adapter.h>
#include <sai/inference/sam2_segmenter.h>
#include <sai/memory/gpu_pool.h>
#include <sai/memory/arena_allocator.h>
#include <sai/image/gpu_image.h>
#include <cuda_runtime.h>
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
#include <sai/pipeline/pipeline.h>
#include <sai/tuning/tuning_space.h>
#include <sai/tuning/tuning_objective.h>
#include <sai/tuning/bayesian_optimizer.h>
#include <sai/tuning/tuning_scheduler.h>
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
    std::string dataset_path;            // --dataset: YAML manifest for coreset building
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
        } else if (arg == "--dataset" && i + 1 < argc) {
            args.dataset_path = argv[++i];
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
    if (!args.dataset_path.empty() && args.coreset_output_path.empty()) {
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
// Uses DINOv3 → PatchEmbedder (1024-dim) with TensorRT GPU inference.
auto BuildCoreset(const CliArgs& cli) -> int {
    using namespace sai;

    std::cout << "Building coreset from dataset: "
              << cli.dataset_path << "\n";

    constexpr std::size_t kEmbedDim = 1024;
    constexpr std::size_t kImageSize = 1024;
    constexpr std::size_t kPatchSize = 14;

    io::BasicImporter importer;

    // ── Create DINOv3 embedder ──
    auto infer_engine = std::make_shared<inference::TensorRtEngine>(
        /*device_ordinal=*/0);
    inference::DinoV3Config dino_cfg;
    dino_cfg.engine_path = "resources/models/dino_v3_vit_base.engine";
    dino_cfg.image_size = kImageSize;
    dino_cfg.patch_size = kPatchSize;
    dino_cfg.embed_dim = kEmbedDim;

    auto dino_adapter = inference::DinoV3Adapter::Create(
        *infer_engine, dino_cfg);
    if (!dino_adapter) {
        std::cerr << "DinoV3Adapter creation failed: "
                  << dino_adapter.error().message << "\n";
        return 1;
    }
    auto patch_emb = embedding::PatchEmbedder::Create(
        std::move(*dino_adapter));
    if (!patch_emb) {
        std::cerr << "PatchEmbedder creation failed\n";
        return 1;
    }
    auto embedder = std::make_unique<embedding::PatchEmbedder>(
        std::move(*patch_emb));
    std::cout << "Embedder: PatchEmbedder (DINOv3, dim="
              << kEmbedDim << ")\n";

    // Read dataset manifest
    auto dataset_result = importer.ImportDataset(cli.dataset_path);
    if (!dataset_result) {
        std::cerr << "Dataset import failed: "
                  << dataset_result.error().message << "\n";
        return 1;
    }
    auto& entries = *dataset_result;
    std::cout << "Found " << entries.size() << " normal sample images\n";

    // Preprocess chain: resize to embedder's expected size
    auto preprocess_chain = image::Compose({
        image::MakeResize(kImageSize, kImageSize)
    });

    // Extract embeddings from each normal image
    std::vector<embedding::Embedding> all_embeddings;
    int processed = 0;

    // GPU upload: SurfaceImage → GpuImage → DINOv3 → DtoH
    sai::memory::ArenaAllocator arena(64 * 1024 * 1024);  // 64 MiB metadata
    auto gpu_pool_result = sai::memory::GpuPool::Create(
        sai::memory::MemoryPoolConfig{
            .slab_size = kImageSize * kImageSize * 3,   // RGB8
            .slab_count = 16,
        },
        arena);
    if (!gpu_pool_result) {
        std::cerr << "GpuPool creation failed: "
                  << gpu_pool_result.error().message << "\n";
        return 1;
    }
    auto& gpu_pool = **gpu_pool_result;

    for (auto& entry : entries) {
        auto img_result = importer.ImportImage(entry.path);
        if (!img_result) {
            std::cerr << "Failed to import " << entry.entry.path.filename().string()
                      << ": " << img_result.error().message << "\n";
            continue;
        }

        // Stamp image metadata from dataset entry
        (*img_result)->Meta().surface_id = entry.surface_id;
        (*img_result)->Meta().position_id = entry.position_id;
        (*img_result)->Meta().light_id = entry.light_id;

        // Run through preprocess (resize) to get SurfaceImage
        auto preprocess_result = preprocess_chain(std::move(*img_result));
        if (!preprocess_result) {
            std::cerr << "Preprocess failed for " << entry.path.filename().string()
                      << ": " << preprocess_result.error().message << "\n";
            continue;
        }

        // Wrap raw Image* as SurfaceImage
        sai::image::SurfaceImage surface = [&]() -> sai::image::SurfaceImage {
            auto* raw = preprocess_result->get();
            if (auto* surf = dynamic_cast<image::SurfaceImage*>(raw)) {
                auto s = std::move(*surf);
                delete raw;  // release unique_ptr ownership
                return s;
            }
            auto meta = raw->Meta();
            const auto* data = raw->Data();
            auto size = raw->SizeBytes();
            std::vector<std::uint8_t> buffer(data, data + size);
            delete raw;
            return sai::image::SurfaceImage::FromOwnedBuffer(
                std::move(buffer), meta);
        }();

        // Extract embedding: HtoD → DINOv3 infer → DtoH
        sai::Result<embedding::Embedding> emb = tl::make_unexpected(
            ErrorInfo{ErrorCode::Inference_EngineExecutionFailed, "no path"});
        {
            // HtoD: upload SurfaceImage to GPU
            auto gpu_img_result = image::GpuImage::FromPool(
                gpu_pool, surface.Meta());
            if (!gpu_img_result) {
                std::cerr << "GpuImage allocation failed for "
                          << entry.path.filename().string() << "\n";
                continue;
            }
            auto gpu_img = std::move(*gpu_img_result);
            auto cuda_err = cudaMemcpy(
                gpu_img.Data(), surface.Data(), surface.SizeBytes(),
                cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                std::cerr << "cudaMemcpy HtoD failed for "
                          << entry.path.filename().string() << "\n";
                continue;
            }

            // Run DINOv3 inference on GPU
            emb = embedder->Extract(gpu_img);
            if (emb && emb->IsOnGpu()) {
                // DtoH: download patch features to CPU for FeatureBank
                auto byte_size = emb->SizeBytes();
                std::vector<float> cpu_data(emb->Meta().count * emb->Meta().dim);
                cuda_err = cudaMemcpy(
                    cpu_data.data(), emb->Data(), byte_size,
                    cudaMemcpyDeviceToHost);
                if (cuda_err != cudaSuccess) {
                    std::cerr << "cudaMemcpy DtoH failed for "
                              << entry.path.filename().string() << "\n";
                    continue;
                }
                emb = embedding::Embedding::FromCpu(
                    std::move(cpu_data), emb->Meta());
            }
        }

        if (!emb) {
            std::cerr << "Embedding extraction failed for "
                      << entry.path.filename().string() << ": "
                      << emb.error().message << "\n";
            continue;
        }

        ++processed;
        all_embeddings.push_back(std::move(*emb));
        std::cout << "[" << processed << "/" << image_files.size() << "] "
                  << entry.path.filename().string() << " → "
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
            emb_ptrs, kEmbedDim, cli.coreset_max_samples);
    } else {
        bank_result = detection::FeatureBank::BuildFromEmbeddings(
            emb_ptrs, kEmbedDim, cli.coreset_max_samples);
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
              << "  Total patches:" << all_embeddings.size() * kEmbedDim << "\n"
              << "  Coreset size: " << bank_result->NumSamples() << "\n"
              << "  Feature dim:  " << bank_result->Dim() << "\n"
              << "  Output:       " << cli.coreset_output_path << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Bayesian Auto-Tuning: YAML helpers for ParameterApplier
// ---------------------------------------------------------------------------

// Collect all leaf nodes from the decision tree YAML in pre-order.
// A leaf node is any YAML mapping with "type: leaf".
static auto CollectLeafNodes(const YAML::Node& node) -> std::vector<YAML::Node> {
    std::vector<YAML::Node> leaves;
    if (!node.IsMap()) return leaves;

    if (auto type = node["type"];
        type.IsDefined() && type.as<std::string>() == "leaf") {
        leaves.push_back(node);
        return leaves;
    }

    if (auto branches = node["branches"];
        branches.IsDefined() && branches.IsMap()) {
        for (auto it = branches.begin(); it != branches.end(); ++it) {
            auto child = CollectLeafNodes(it->second);
            leaves.insert(leaves.end(),
                          std::make_move_iterator(child.begin()),
                          std::make_move_iterator(child.end()));
        }
    }

    if (auto def = node["default"]; def.IsDefined()) {
        auto child = CollectLeafNodes(def);
        leaves.insert(leaves.end(),
                      std::make_move_iterator(child.begin()),
                      std::make_move_iterator(child.end()));
    }

    return leaves;
}

// Apply a single parameter value to the YAML tree.
// param_name formats:
//   "verdict_mapping.X"        -> root["verdict_mapping"][X] = value
//   "leaf_N.formula_M.weight_K" -> leaf_nodes[N]["formulas"][M]["weights"][K] = value
//   "leaf_N.formula_M.threshold" -> leaf_nodes[N]["formulas"][M]["threshold"] = value
static auto ApplyParamToYaml(YAML::Node& root,
                              std::vector<YAML::Node>& leaf_nodes,
                              const std::string& param_name,
                              double value) -> void {
    std::vector<std::string> segs;
    {
        std::istringstream iss(param_name);
        std::string seg;
        while (std::getline(iss, seg, '.')) segs.push_back(seg);
    }
    if (segs.empty()) return;

    if (segs[0] == "verdict_mapping") {
        if (segs.size() == 2) {
            root["verdict_mapping"][segs[1]] = value;
        }
    } else if (segs.size() >= 3 && segs[0].rfind("leaf_", 0) == 0) {
        try {
            size_t li = std::stoul(segs[0].substr(5));
            if (li >= leaf_nodes.size()) return;
            auto& leaf = leaf_nodes[li];

            if (segs[1].rfind("formula_", 0) == 0) {
                size_t fi = std::stoul(segs[1].substr(8));
                if (segs[2] == "threshold") {
                    leaf["formulas"][fi]["threshold"] = value;
                } else if (segs.size() == 4 && segs[2].rfind("weight_", 0) == 0) {
                    size_t wi = std::stoul(segs[2].substr(7));
                    leaf["formulas"][fi]["weights"][wi] = value;
                }
            }
        } catch (...) {
            // Invalid segment — silently ignore
        }
    }
}

}  // anonymous namespace

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    using namespace sai;

    // =========================================================================
    // --dataset mode: build FeatureBank from normal sample dataset YAML
    // =========================================================================
    if (!cli.dataset_path.empty()) {
        return BuildCoreset(cli);
    }

    // =========================================================================
    // 1. DI container
    // =========================================================================
    auto ctx = std::make_unique<Context>();
    (void)ctx->Initialize();
    (void)ctx->Start();

    // =========================================================================
    // 2. Inference / embedding setup — TensorRT + DINOv3 (1024-dim)
    // =========================================================================
    constexpr std::size_t kEmbedDim = 1024;

    // Primary: DINOv3 PatchEmbedder
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
        return 1;
    }
    auto patch_emb = embedding::PatchEmbedder::Create(std::move(*dino_adapter));
    if (!patch_emb) {
        std::cerr << "PatchEmbedder creation failed\n";
        return 1;
    }
    auto embedder = std::make_shared<embedding::PatchEmbedder>(std::move(*patch_emb));
    std::cout << "Embedder: PatchEmbedder (DINOv3)\n";

    // Global embedder (CLIP) for cross-modal vector retrieval.
    // Enabled via pipeline.yaml → stages[2].config.global_model.enabled
    std::shared_ptr<embedding::IEmbedder> global_embedder;
    {
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
        auto global_cfg = pipeline_yaml["pipeline"]["stages"][2]["config"]["global_model"];
        if (global_cfg.IsDefined() && global_cfg["enabled"].as<bool>(false)) {
            auto clip_engine = std::make_shared<inference::TensorRtEngine>(
                /*device_ordinal=*/0);
            inference::ClipConfig clip_cfg;
            clip_cfg.engine_path = "resources/models/clip_vit_b32.engine";
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
    // 4b. Coreset Self-Evolution — YAML config-driven
    // =========================================================================
    std::unique_ptr<detection::CoresetEvolution> evolution;
    if (feature_bank) {
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
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
                    evolution = std::make_unique<detection::CoresetEvolution>(
                        std::move(*evo_cfg), *patch_core, std::move(*profile));
                    std::cout << "CoresetEvolution: enabled (target_size="
                              << evo_cfg->target_size << ")\n";
                }
            }
        }
    }

    // =========================================================================
    // 5. KnowledgeStore — unified KG + Evolution facade (replaces raw sqlite3 + KG)
    // =========================================================================
    auto ks_result = knowledge::KnowledgeStore::Create(
        knowledge::KnowledgeStore::Config{":memory:", true});
    if (!ks_result) {
        std::cerr << "KnowledgeStore creation failed: "
                  << ks_result.error().message << "\n";
        return 1;
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
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
        auto sam2_cfg = pipeline_yaml["pipeline"]["stages"][5]["config"]["sam2"];
        if (sam2_cfg.IsDefined() && sam2_cfg["enabled"].as<bool>(false)) {
            auto sam2_engine = std::make_shared<inference::TensorRtEngine>(
                /*device_ordinal=*/0);
            inference::Sam2Config s2_cfg;
            s2_cfg.engine_path = "resources/models/sam2_vit_h.engine";
            s2_cfg.image_size = 1024;

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
    std::unique_ptr<sai::tuning::TuningScheduler> tuning_scheduler;
    std::stop_source tuning_stop_source;

    {
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
        auto tuning_node = pipeline_yaml["pipeline"]["tuning"];

        if (tuning_node.IsDefined() && tuning_node["enabled"].as<bool>(false)) {
            auto resource_dir = std::filesystem::path("resources");
            auto tuning_cfg_path = resource_dir / "tuning" / "seat_leather_tuning.yaml";

            // Load tuning YAML for all config sections
            auto tuning_yaml = YAML::LoadFile(tuning_cfg_path.string());
            auto ty = tuning_yaml["tuning"];

            // 1. Parse TuningSpace
            auto space_result = sai::tuning::TuningSpace::LoadFromYaml(tuning_cfg_path);
            if (!space_result.has_value()) {
                std::cerr << "Tuning: failed to load tuning space: "
                          << space_result.error().message << "\n";
            } else {
                auto space = std::move(*space_result);
                auto space_dim = space.Dimension();

                // Capture parameter names before space is moved into BayesianOptimizer
                std::vector<std::string> param_names;
                for (const auto& p : space.Parameters()) {
                    param_names.push_back(p.name);
                }

                // 2. Parse SchedulerConfig from tuning YAML
                sai::tuning::SchedulerConfig sched_cfg;
                if (auto sn = ty["scheduler"]; sn.IsDefined()) {
                    sched_cfg.interval = std::chrono::seconds(
                        sn["interval_sec"].as<int>(3600));
                    sched_cfg.monitoring_window = std::chrono::seconds(
                        sn["monitoring_window_sec"].as<int>(300));
                    sched_cfg.feedback_lookback = std::chrono::seconds(
                        sn["feedback_lookback_sec"].as<int>(86400));
                }
                if (auto sn = ty["safety"]; sn.IsDefined()) {
                    sched_cfg.min_ng_rate = sn["min_ng_rate"].as<double>(0.001);
                    sched_cfg.max_ng_rate = sn["max_ng_rate"].as<double>(0.50);
                    sched_cfg.min_samples_for_trigger =
                        sn["min_samples_for_trigger"].as<std::size_t>(50);
                }

                // 3. Parse OptimizerConfig from tuning YAML
                sai::tuning::OptimizerConfig opt_cfg;
                if (auto on = ty["optimizer"]; on.IsDefined()) {
                    opt_cfg.max_iterations =
                        on["max_iterations"].as<std::size_t>(50);
                    opt_cfg.initial_random_points =
                        on["initial_random_points"].as<std::size_t>(5);
                    opt_cfg.noise_level = on["noise_level"].as<double>(0.01);
                }

                // 4. Create components
                double fp_cost = ty["objective"]["fp_cost"].as<double>(1.0);
                double fn_cost = ty["objective"]["fn_cost"].as<double>(5.0);
                auto objective = std::make_unique<sai::tuning::KnowledgeGraphObjective>(
                    ks->Graph(), fp_cost, fn_cost);

                auto optimizer = std::make_unique<sai::tuning::BayesianOptimizer>(
                    std::move(space), opt_cfg);

                tuning_scheduler = std::make_unique<sai::tuning::TuningScheduler>(
                    sched_cfg,
                    std::move(optimizer),
                    std::move(objective),
                    kg,
                    kg_evolution);

                // 5. Inject callbacks
                auto tree_path = resource_dir / "trees" / "seat_leather_inspection.yaml";
                tuning_scheduler->SetParameterApplier(
                    [&reasoner, tree_path, param_names = std::move(param_names)]
                    (const std::vector<double>& params) -> Result<void> {
                        // 1. Read current YAML
                        YAML::Node root;
                        try {
                            root = YAML::LoadFile(tree_path.string());
                        } catch (const YAML::Exception& e) {
                            return tl::make_unexpected(ErrorInfo{
                                ErrorCode::Pipeline_InvalidConfig,
                                "Tuning: failed to load decision tree YAML for parameter update: " +
                                    std::string(e.what())});
                        }

                        // 2. Collect leaf nodes in pre-order
                        auto leaf_nodes = CollectLeafNodes(root);

                        // 3. Apply each parameter value to its YAML path
                        for (size_t i = 0; i < params.size() && i < param_names.size(); ++i) {
                            ApplyParamToYaml(root, leaf_nodes, param_names[i], params[i]);
                        }

                        // 4. Write to temp file, then atomically rename
                        auto tmp = tree_path.string() + ".tmp";
                        {
                            std::ofstream out(tmp);
                            out << root;
                        }
                        std::filesystem::rename(tmp, tree_path);

                        // 5. Reload tree with updated parameters
                        return reasoner->ReloadTree(tree_path);
                    });

                tuning_scheduler->SetMetricsPoller(
                    [&pipeline]() -> Result<sai::tuning::MetricsSnapshot> {
                        sai::tuning::MetricsSnapshot snapshot;
                        auto metrics = pipeline->Metrics();
                        for (const auto& sm : metrics) {
                            // Use the Reason stage's metrics to compute NG rate.
                            // frames_failed on the Reason stage captures processing
                            // failures from the decision tree evaluation.
                            if (sm.type == sai::pipeline::StageType::Reason) {
                                auto processed = sm.frames_processed.load();
                                auto failed = sm.frames_failed.load();
                                snapshot.sample_count += processed;
                                if (processed > 0) {
                                    snapshot.ng_rate +=
                                        static_cast<double>(failed) / processed * processed;
                                }
                            }
                        }
                        if (snapshot.sample_count > 0) {
                            snapshot.ng_rate /= snapshot.sample_count;
                        }
                        return snapshot;
                    });

                // 6. Start tuning thread
                tuning_scheduler->Start(tuning_stop_source.get_token());

                std::cout << "TuningScheduler: started with " << space_dim
                          << " parameters\n";
            }
        }
    }

    // =========================================================================
    // 9b. Result callback for self-evolution (shared by headless and GUI modes)
    // =========================================================================
    if (evolution) {
        pipeline->SetResultCallback(
            [&](int fid, const sai::reasoner::ReasoningResult& result) {
                (void)fid;
                if (evolution && evolution->IsRunning()) {
                    const auto& ctx = patch_core->LastContext();
                    if (!ctx.knn_distances.empty()) {
                        evolution->AssessAndOffer(
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
        std::cerr << "Pipeline start failed: "
                  << start_result.error().message << "\n";
        return 1;
    }

    // =========================================================================
    // 10b. Start self-evolution background thread
    // =========================================================================
    if (evolution) {
        std::stop_source evo_ss;
        evolution->Start(evo_ss.get_token());
        std::cout << "CoresetEvolution: started\n";
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
                      << entry.path.filename().string() << " → " << verdict << "\n";
        }

        std::cout << "\n===== Summary =====\n"
                  << "Total:   " << image_files.size() << "\n"
                  << "OK:      " << ok << "\n"
                  << "NG:      " << ng << "\n"
                  << "WARN:    " << warn << "\n"
                  << "UNCERTAIN: " << uncertain << "\n"
                  << "Failed:  " << failed << "\n"
                  << "Results: " << cli.output_dir << "\n";

        if (evolution) evolution->Stop();
        if (tuning_scheduler) tuning_scheduler->Join();
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

    if (!evolution) {
        // No evolution — keep simple callback (no self-evolution capture needed)
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
    } else {
        pipeline->SetResultCallback(
            [&](int fid, const reasoner::ReasoningResult& result) {
                inspection_vm->UpdateResult(fid, result);
                visualization::FrameSummary summary;
                summary.frame_id = fid;
                summary.verdict = result.verdict;
                summary.severity = std::to_string(result.severity);
                summary.timestamp = std::chrono::system_clock::now();
                dashboard_vm->AppendFrameSummary(std::move(summary));

                if (evolution && evolution->IsRunning()) {
                    const auto& ctx = patch_core->LastContext();
                    if (!ctx.knn_distances.empty()) {
                        evolution->AssessAndOffer(
                            ctx.knn_distances.data(),
                            ctx.knn_distances.size() / ctx.k_nearest,
                            ctx.k_nearest,
                            ctx.embedding_data.data(),
                            ctx.grid_h,
                            ctx.grid_w,
                            ctx.dim,
                            ctx.detection_result,
                            result.triggered_rules.size(),
                            result.verdict,
                            ctx.effective_threshold,
                            ctx.pca_image_score,
                            ctx.pca_self_query_p95);
                    }
                }
            });
    }

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
        if (evolution) evolution->Stop();
        if (tuning_scheduler) tuning_scheduler->Join();
        (void)pipeline->Stop();
        (void)ctx->Stop();
    });

    return app.exec();
}
