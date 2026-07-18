#pragma once

#include <map>
#include <memory>
#include <optional>
#include <stop_token>

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
namespace memory { class GpuPool; }
}

struct CliArgs;

struct AssembledApp {
    AssembledApp() = default;
    ~AssembledApp();  // out-of-line: unique_ptr safe against incomplete types
    AssembledApp(AssembledApp&&) noexcept = default;  // inline: all types complete in header

    // ── Core (must exist) ──
    std::unique_ptr<sai::Context> ctx;
    std::unique_ptr<sai::pipeline::Pipeline> pipeline;
    std::shared_ptr<sai::embedding::PatchEmbedder> embedder;
    std::shared_ptr<sai::detection::PatchCore> patch_core;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine;
    std::shared_ptr<sai::io::JsonExporter> exporter;

    // ── Knowledge (kg/evolution are non-owning aliases into knowledge_store) ──
    std::unique_ptr<sai::knowledge::KnowledgeStore> knowledge_store;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg;
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> kg_evolution;

    // ── Optional components ──
    std::shared_ptr<sai::embedding::IEmbedder> global_embedder;
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner;
    std::shared_ptr<sai::detection::FeatureBank> feature_bank;
    std::shared_ptr<sai::retrieval::VectorPath> vector_path;

    // unique_ptr: CoresetEvolution/TuningScheduler are immovable (move=delete),
    // so optional<T> can't hold them. unique_ptr works because we move the pointer.
    std::unique_ptr<sai::detection::CoresetEvolution> evolution;
    std::unique_ptr<sai::tuning::TuningScheduler> tuning_scheduler;
    std::shared_ptr<sai::memory::GpuPool> gpu_pool;
    std::stop_source tuning_stop_source;
    std::stop_source evolution_stop_source;

    // ── Multi-position detectors ──
    using BankKey = sai::pipeline::BankKey;
    std::map<BankKey, std::shared_ptr<sai::detection::PatchCore>> patch_cores;
    std::map<BankKey, std::unique_ptr<sai::detection::CoresetEvolution>> evolutions;
    std::map<BankKey, std::stop_source> evolution_stop_sources;
};

auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>;
