#pragma once

#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <vector>

#include <sai/core/error.h>
#include <sai/detection/coreset_evolution.h>  // complete type for map value
#include <sai/pipeline/pipeline_config.h>     // BankKey
#include <sai/tuning/tuning_scheduler.h>      // complete type for unique_ptr

// Forward declarations — shared_ptr members are fine with forward decls
// (type-erased deleter). unique_ptr members need the destructor out-of-line
// (see app_builder.cpp).
namespace sai {
class Context;
namespace pipeline { class Pipeline; }
namespace embedding { class PatchEmbedder; class IEmbedder; }
namespace detection { class PatchCore; class FeatureBank; }
namespace rule { class RuleEngine; }
namespace reasoner { class IReasoner; }
namespace io { class JsonExporter; }
namespace knowledge { class KnowledgeGraph; class KnowledgeEvolution; class KnowledgeStore; }
namespace inference { class Sam2Segmenter; }
namespace retrieval { class VectorPath; }
namespace memory { class ArenaAllocator; class GpuPool; }
}

struct CliArgs;

// ── Per-position pipeline instance ──
// Each camera position gets its own independent 7-stage pipeline,
// its own PatchCore + FeatureBank, and its own CoresetEvolution.
// Shared components (RuleEngine, Reasoner, KnowledgeStore, etc.) are
// injected into each pipeline's stages.
struct PositionPipeline {
    PositionPipeline() = default;
    ~PositionPipeline();  // out-of-line: unique_ptr safe against incomplete types
    PositionPipeline(PositionPipeline&&) noexcept = default;

    using BankKey = sai::pipeline::BankKey;
    BankKey key;

    std::unique_ptr<sai::pipeline::Pipeline> pipeline;
    std::shared_ptr<sai::detection::PatchCore> patch_core;
    std::unique_ptr<sai::memory::ArenaAllocator> image_pool_arena;
    std::unique_ptr<sai::memory::ArenaAllocator> embedding_pool_arena;
    std::shared_ptr<sai::memory::GpuPool> image_pool;
    std::shared_ptr<sai::memory::GpuPool> embedding_pool;
    std::shared_ptr<sai::embedding::PatchEmbedder> embedder;
    std::shared_ptr<sai::embedding::IEmbedder> global_embedder;
    std::unique_ptr<sai::detection::CoresetEvolution> evolution;
    std::stop_source evolution_stop_source;
};

// ── Assembled application ──
// Shared components are created once and injected into each PositionPipeline.
struct AssembledApp {
    AssembledApp() = default;
    ~AssembledApp();  // out-of-line: unique_ptr safe against incomplete types
    AssembledApp(AssembledApp&&) noexcept = default;  // inline: all types complete in header

    // ── Shared components ──
    std::unique_ptr<sai::Context> ctx;
    std::unique_ptr<sai::knowledge::KnowledgeStore> knowledge_store;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg;            // non-owning alias
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> kg_evolution; // non-owning alias
    std::shared_ptr<sai::rule::RuleEngine> rule_engine;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner;
    std::shared_ptr<sai::io::JsonExporter> exporter;
    std::shared_ptr<sai::retrieval::VectorPath> vector_path;
    std::shared_ptr<sai::detection::FeatureBank> feature_bank;    // single-position coreset

    // ── Optional shared ──
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter;
    std::unique_ptr<sai::tuning::TuningScheduler> tuning_scheduler;
    std::stop_source tuning_stop_source;

    // ── Per-position pipelines ──
    std::vector<PositionPipeline> positions;

    // ── Legacy convenience accessors (single-position mode) ──
    [[nodiscard]] auto GetPipeline() -> sai::pipeline::Pipeline& { return *positions.at(0).pipeline; }
    [[nodiscard]] auto GetPatchCore() -> sai::detection::PatchCore& { return *positions.at(0).patch_core; }
    [[nodiscard]] auto HasPosition(const std::string& sid, std::uint16_t pid) const -> bool;
    [[nodiscard]] auto FindPosition(const std::string& sid, std::uint16_t pid) -> PositionPipeline*;
};

auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>;
