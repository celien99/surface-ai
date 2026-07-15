#pragma once

#include <map>
#include <memory>
#include <optional>
#include <stop_token>

#include <sai/core/error.h>

// Forward declarations
namespace sai {
class Context;
namespace pipeline { class Pipeline; }
namespace embedding { class PatchEmbedder; class IEmbedder; }
namespace detection { class PatchCore; class FeatureBank; struct CoresetEvolution; }
namespace rule { class RuleEngine; }
namespace reasoner { class IReasoner; }
namespace io { class JsonExporter; }
namespace knowledge { class KnowledgeGraph; class KnowledgeEvolution; class KnowledgeStore; }
namespace inference { class Sam2Segmenter; }
namespace retrieval { class VectorPath; }
namespace tuning { class TuningScheduler; }
}

struct CliArgs;

struct AssembledApp {
    // ── Core (must exist) ──
    std::unique_ptr<sai::Context> ctx;
    std::unique_ptr<sai::pipeline::Pipeline> pipeline;
    std::shared_ptr<sai::embedding::PatchEmbedder> embedder;
    std::shared_ptr<sai::detection::PatchCore> patch_core;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine;
    std::shared_ptr<sai::io::JsonExporter> exporter;

    // ── Knowledge ──
    std::unique_ptr<sai::knowledge::KnowledgeStore> knowledge_store;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg;
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> kg_evolution;

    // ── Optional components ──
    std::shared_ptr<sai::embedding::IEmbedder> global_embedder;
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner;
    std::shared_ptr<sai::detection::FeatureBank> feature_bank;
    std::shared_ptr<sai::retrieval::VectorPath> vector_path;

    std::optional<sai::detection::CoresetEvolution> evolution;
    std::optional<sai::tuning::TuningScheduler> tuning_scheduler;
    std::stop_source tuning_stop_source;    // keeps tuning thread alive
    std::stop_source evolution_stop_source;  // keeps evolution thread alive

    // ── Multi-position detectors ──
    using BankKey = std::pair<std::string, std::uint16_t>;
    // Ownership: PatchCore owns FeatureBank.
    // Declared BEFORE evolutions so destruction order is evolutions → patch_cores
    std::map<BankKey, std::shared_ptr<sai::detection::PatchCore>> patch_cores;
    // Each evolution holds a PatchCore& (non-owning). Declared AFTER patch_cores.
    std::map<BankKey, sai::detection::CoresetEvolution> evolutions;
    std::map<BankKey, std::stop_source> evolution_stop_sources;
};

auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>;
