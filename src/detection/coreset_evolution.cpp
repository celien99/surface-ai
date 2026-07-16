// coreset_evolution.cpp — CoresetEvolution 门面（PIMPL）
#include <sai/detection/coreset_evolution.h>
#include <sai/detection/detection_result.h>
#include <sai/knowledge/knowledge_store.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedding.h>
#include <sai/infra/logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::detection {
namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

constexpr std::size_t kDriftHistoryWindowSize = 5;
constexpr double kDriftSigmaMultiplier = 2.0;

// ── Internal helper: compute normalcy score from distances ──
auto ComputeNormalcy(const float* distances, std::size_t count,
                     const NormalityProfile& profile, float tail_ratio_max)
    -> NormalityAssessment {
    if (count == 0 || profile.num_samples == 0) return {};

    std::vector<float> sorted(distances, distances + count);
    auto mid = sorted.begin() + static_cast<std::ptrdiff_t>(count / 2);
    std::nth_element(sorted.begin(), mid, sorted.end());
    float median_dist = *mid;

    float concentration = (profile.p50 > 0.0F) ? median_dist / profile.p50 : 1.0F;

    std::size_t tail = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (distances[i] > profile.p95) ++tail;
    }
    float tail_ratio = static_cast<float>(tail) / static_cast<float>(count);

    float score = 1.0F;
    if (tail_ratio > 0.0F && tail_ratio_max > 0.0F) {
        score = 1.0F - std::min(1.0F, tail_ratio / tail_ratio_max);
    }

    return {score, concentration, tail_ratio};
}

// ── Internal: novelty check ──
auto CheckNovelty(const float* distances, std::size_t count,
                  const NormalityProfile& profile, float coverage_threshold) -> NoveltyResult {
    NoveltyResult r;
    if (count == 0) return r;
    std::size_t covered = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (distances[i] < profile.p50) ++covered;
    }
    r.coverage_ratio = static_cast<float>(covered) / static_cast<float>(count);
    r.novel_patch_count = count - covered;
    r.is_novel = (r.coverage_ratio < coverage_threshold);
    return r;
}

// ── Internal: light greedy prefilter ──
auto PrefilterCandidates(const std::vector<float>& vectors, std::size_t dim,
                         std::size_t target) -> std::vector<float> {
    auto total = vectors.size() / dim;
    if (total <= target) return vectors;

    std::vector<float> out;
    out.reserve(target * dim);
    auto stride = total / target;
    if (stride < 1) stride = 1;
    for (std::size_t i = 0; i < target && i * stride < total; ++i) {
        auto idx = i * stride;
        out.insert(out.end(),
                   vectors.begin() + static_cast<std::ptrdiff_t>(idx * dim),
                   vectors.begin() + static_cast<std::ptrdiff_t>((idx + 1) * dim));
    }
    return out;
}

}  // namespace

// ── EvolutionConfig::FromYaml ─────────────────────────────────────

auto EvolutionConfig::FromYaml(const YAML::Node& node) -> Result<EvolutionConfig> {
    EvolutionConfig cfg;
    try {
        auto se = node["self_evolution"];
        if (!se.IsDefined()) {
            cfg.enabled = false;
            return cfg;
        }

        cfg.enabled = se["enabled"].as<bool>(false);

        if (auto n = se["normality"]; n.IsDefined()) {
            cfg.normality_k = n["k_self_query"].as<std::size_t>(5);
            cfg.tail_ratio_max = n["tail_ratio_max"].as<float>(0.10F);
        }

        if (auto n = se["novelty"]; n.IsDefined()) {
            cfg.coverage_threshold = n["coverage_threshold"].as<float>(0.60F);
        }

        if (auto b = se["buffer"]; b.IsDefined()) {
            cfg.max_frames = b["max_frames"].as<std::size_t>(50);
            cfg.max_patches = b["max_patches"].as<std::size_t>(50000);
            cfg.trigger_frames = b["trigger_frames"].as<std::size_t>(20);
            cfg.trigger_patches = b["trigger_patches"].as<std::size_t>(20000);
        }

        if (auto u = se["update"]; u.IsDefined()) {
            cfg.target_size = u["target_size"].as<std::size_t>(10000);
            cfg.min_update_interval = std::chrono::seconds{
                u["min_interval_sec"].as<int>(5)};
            cfg.greedy_prefilter = u["greedy_prefilter"].as<std::size_t>(5000);
        }

        if (auto p = se["persistence"]; p.IsDefined()) {
            cfg.save_on_stop = p["save_on_stop"].as<bool>(true);
            cfg.backup_old_bank = p["backup_old_bank"].as<bool>(true);
            cfg.max_backups = p["max_backups"].as<std::size_t>(3);
        }
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "self_evolution config parse error: " + std::string(e.what()),
            std::source_location::current(),
        });
    }
    return cfg;
}

struct CoresetEvolution::Impl {
    EvolutionConfig cfg;
    PatchCore& detector;
    NormalityProfile active_profile;
    NormalityProfile standby_profile;
    CandidateBuffer buffer;
    EvolutionStats stats;
    std::unique_ptr<FeatureBank> standby_bank;

    // Background thread
    std::jthread update_thread;
    std::atomic<bool> notify_flag{false};
    std::mutex cv_mutex;  // required by condition_variable_any (CV protocol)
    std::condition_variable_any notify_cv;
    std::mutex stats_mutex;

    std::shared_ptr<knowledge::KnowledgeStore> knowledge_store;
    std::filesystem::path save_path;

    // Sliding window for drift detection
    std::vector<float> displacement_history;

    auto CheckDrift(float new_displacement) -> void {
        displacement_history.push_back(new_displacement);
        if (displacement_history.size() > kDriftHistoryWindowSize) {
            displacement_history.erase(displacement_history.begin());
        }
        if (displacement_history.size() < 3) return;

        // Compute mean + stddev of history
        double sum = 0.0;
        for (auto d : displacement_history) sum += static_cast<double>(d);
        double mean = sum / static_cast<double>(displacement_history.size());
        double var = 0.0;
        for (auto d : displacement_history) {
            double diff = static_cast<double>(d) - mean;
            var += diff * diff;
        }
        double stddev = std::sqrt(var / static_cast<double>(displacement_history.size()));

        // Check if last 3 are all > mean + 2*stddev
        if (stddev > 0.0) {
            auto threshold = mean + kDriftSigmaMultiplier * stddev;
            auto n = displacement_history.size();
            if (displacement_history[n-1] > threshold
                && displacement_history[n-2] > threshold
                && displacement_history[n-3] > threshold) {
                sai::infra::Logger::Get("detection").Log(sai::infra::LogLevel::Warning,
                    "[CoresetEvolution] Potential concept drift: "
                    "mean_displacement={} exceeds 2σ threshold={}. "
                    "Self-evolution continues.",
                    new_displacement, threshold);
            }
        }
    }

    Impl(EvolutionConfig c, PatchCore& d, NormalityProfile p)
        : cfg(std::move(c))
        , detector(d)
        , active_profile(std::move(p))
        , standby_profile(active_profile)
        , buffer(CandidateBuffer::Config{
              cfg.max_frames, cfg.max_patches,
              cfg.trigger_frames, cfg.trigger_patches})
    {
        InitializeStandbyBank();
    }

    // Clone PatchCore's active bank as the initial standby.
    // Uses temporary SwapFeatureBank(nullptr) to extract, clone, then restore.
    void InitializeStandbyBank() {
        if (active_profile.dim == 0) return;

        auto active = detector.SwapFeatureBank(nullptr);
        if (!active || active->NumSamples() == 0) {
            std::ignore = detector.SwapFeatureBank(std::move(active));
            return;
        }

        auto vecs = active->ExtractAllVectors();
        auto dim = active->Dim();
        auto count = active->NumSamples();

        // Restore active bank immediately (returned nullptr, discard)
        std::ignore = detector.SwapFeatureBank(std::move(active));

        // Build standby clone from extracted vectors
        sai::embedding::EmbeddingMeta meta;
        meta.model_name = "standby_clone";
        meta.type = sai::embedding::EmbeddingType::Patch;
        meta.dim = dim;
        meta.count = count;
        meta.grid = {count, 1};
        auto emb = sai::embedding::Embedding::FromCpu(std::move(vecs), std::move(meta));
        std::vector<const sai::embedding::Embedding*> ptrs{&emb};
        auto fb_result = FeatureBank::BuildFromEmbeddings(ptrs, dim, count);
        if (fb_result.has_value()) {
            standby_bank = std::make_unique<FeatureBank>(std::move(*fb_result));
        }
    }
};

CoresetEvolution::CoresetEvolution(EvolutionConfig cfg,
                                   PatchCore& detector,
                                   NormalityProfile profile) noexcept
    : impl_(std::make_unique<Impl>(std::move(cfg), detector, std::move(profile))) {}

CoresetEvolution::~CoresetEvolution() = default;

auto CoresetEvolution::AssessAndOffer(
    const float* distances,
    std::size_t query_count,
    std::size_t /*k*/,
    std::shared_ptr<const std::vector<float>> embedding_data,
    std::size_t grid_h,
    std::size_t grid_w,
    std::size_t dim,
    const DetectionResult& det_result,
    std::size_t matched_rules_count,
    const std::string& reasoner_verdict,
    float effective_threshold,
    float pca_image_score,
    float pca_self_query_p95) noexcept -> void {

    if (!impl_->cfg.enabled) return;
    if (query_count == 0) return;
    if (!embedding_data) return;
    if (embedding_data->empty()) return;

    try {
        // 1. Normalcy score
        auto normalcy = ComputeNormalcy(
            distances, query_count, impl_->active_profile, impl_->cfg.tail_ratio_max);

        // 2. Multi-signal consensus
        bool consensus = MultiSignalConsensusCheck(
            normalcy, det_result, matched_rules_count, reasoner_verdict,
            effective_threshold, pca_image_score, pca_self_query_p95);

        if (!consensus) return;

        // 3. Novelty filter
        auto novelty = CheckNovelty(
            distances, query_count, impl_->active_profile, impl_->cfg.coverage_threshold);

        if (!novelty.is_novel) return;

        // 4. Append to buffer — zero-copy: reuse the shared_ptr from Detect().
        EvolutionCandidate candidate;
        candidate.patch_vectors = std::shared_ptr<const float>(
            embedding_data, embedding_data->data());
        candidate.grid_h = grid_h;
        candidate.grid_w = grid_w;
        candidate.dim = dim;
        candidate.normalcy_score = normalcy.normalcy_score;
        candidate.captured_at = std::chrono::steady_clock::now();

        if (impl_->buffer.Append(std::move(candidate)) && impl_->buffer.IsTriggered()) {
            // Wake up background thread (lock-free — atomic store + notify)
            impl_->notify_flag.store(true, std::memory_order_release);
            impl_->notify_cv.notify_one();
        }
    } catch (...) {
        // Hot path — never let self-evolution crash detection
        sai::infra::Logger::Get("detection").Log(sai::infra::LogLevel::Error,
                                                  "CoresetEvolution::AssessAndOffer exception");
    }
}

auto CoresetEvolution::Start(std::stop_token token) noexcept -> void {
    if (!impl_->cfg.enabled) return;

    impl_->update_thread = std::jthread([this](std::stop_token tok) {
        while (!tok.stop_requested()) {
            {
                std::unique_lock lock(impl_->cv_mutex);
                impl_->notify_cv.wait_for(lock, tok, 500ms, [this, &tok] {
                    return impl_->notify_flag.load(std::memory_order_acquire)
                           || tok.stop_requested();
                });
            }
            if (tok.stop_requested()) break;
            impl_->notify_flag.store(false, std::memory_order_release);

            // Drain + prefilter + merge + greedy coreset + swap
            auto candidates = impl_->buffer.DrainAll();
            if (candidates.empty()) continue;

            auto start = std::chrono::steady_clock::now();

            // Flatten candidate patches
            std::vector<float> candidate_vecs;
            std::size_t dim = impl_->active_profile.dim;
            for (auto& c : candidates) {
                auto n = c.grid_h * c.grid_w * c.dim;
                candidate_vecs.insert(candidate_vecs.end(),
                                      c.patch_vectors.get(),
                                      c.patch_vectors.get() + n);
            }

            // Prefilter
            auto prefiltered = PrefilterCandidates(
                candidate_vecs, dim, impl_->cfg.greedy_prefilter);

            // Merge with existing coreset (standby bank)
            if (!impl_->standby_bank) continue;

            auto existing = impl_->standby_bank->ExtractAllVectors();
            existing.insert(existing.end(), prefiltered.begin(), prefiltered.end());

            // Greedy coreset reselect
            auto total_count = existing.size() / dim;
            auto target = std::min(impl_->cfg.target_size, total_count);

            std::vector<float> selected;
            if (total_count <= target) {
                selected = std::move(existing);
            } else {
                sai::embedding::EmbeddingMeta meta;
                meta.model_name = "merged_update";
                meta.type = sai::embedding::EmbeddingType::Patch;
                meta.dim = dim;
                meta.count = total_count;
                meta.grid = {total_count, 1};
                auto emb = sai::embedding::Embedding::FromCpu(
                    std::move(existing), std::move(meta));
                std::vector<const sai::embedding::Embedding*> ptrs{&emb};
                auto fb = FeatureBank::BuildWithGreedyCoreset(ptrs, dim, target);
                if (fb.has_value()) {
                    selected = fb->ExtractAllVectors();
                }
            }

            if (selected.empty()) continue;

            // Build into a new FeatureBank
            {
                sai::embedding::EmbeddingMeta meta;
                meta.model_name = "updated";
                meta.type = sai::embedding::EmbeddingType::Patch;
                meta.dim = dim;
                meta.count = selected.size() / dim;
                meta.grid = {meta.count, 1};
                std::vector<float> sel_copy = selected;
                auto emb = sai::embedding::Embedding::FromCpu(
                    std::move(sel_copy), std::move(meta));
                std::vector<const sai::embedding::Embedding*> ptrs{&emb};
                auto fb = FeatureBank::BuildFromEmbeddings(ptrs, dim, meta.count);
                if (!fb.has_value()) continue;
                auto new_bank = std::make_unique<FeatureBank>(std::move(*fb));

                // Swap into PatchCore
                auto old_bank = impl_->detector.SwapFeatureBank(std::move(new_bank));
                impl_->standby_bank = std::move(old_bank);

                // Update profile (fast sampling)
                impl_->standby_profile = std::move(impl_->active_profile);
                impl_->active_profile = NormalityProfile::ComputeFast(
                    *impl_->standby_bank, impl_->cfg.normality_k);
            }

            // Record stats
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);

            {
                std::lock_guard lock(impl_->stats_mutex);
                impl_->stats.frames_added = candidates.size();
                impl_->stats.update_duration = elapsed;
                impl_->stats.update_count++;
            }

            // Record evolution event to KnowledgeStore
            if (impl_->knowledge_store) {
                sai::knowledge::KnowledgeRecord props;
                props.fields["event_type"] = sai::knowledge::FieldValue{
                    std::string{"RuntimeUpdate"}};
                props.fields["frames_added"] = sai::knowledge::FieldValue{
                    static_cast<std::int64_t>(candidates.size())};
                props.fields["mean_displacement"] = sai::knowledge::FieldValue{
                    0.0};  // computed in FullRebuild, skipped for runtime
                props.fields["update_duration_ms"] = sai::knowledge::FieldValue{
                    static_cast<std::int64_t>(elapsed.count())};

                auto node_result = impl_->knowledge_store->InsertNode(
                    "CoresetEvolutionEvent", std::move(props));
                // Best-effort — failure doesn't block evolution
                (void)node_result;
            }

            // Concept drift check (independent of knowledge store)
            impl_->CheckDrift(0.0F);
        }
    });
}

auto CoresetEvolution::Stop() noexcept -> void {
    if (impl_->update_thread.joinable()) {
        impl_->update_thread.request_stop();
        impl_->notify_flag.store(true, std::memory_order_release);
        impl_->notify_cv.notify_one();
        impl_->update_thread.join();
    }

    // FullRebuild on stop if enabled and path is set
    if (impl_->cfg.save_on_stop && !impl_->save_path.empty()) {
        std::ignore = FullRebuild(impl_->save_path);
    }
}

auto CoresetEvolution::IsRunning() const noexcept -> bool {
    return impl_->update_thread.joinable();
}

auto CoresetEvolution::LatestStats() const noexcept -> EvolutionStats {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}

auto CoresetEvolution::Profile() const noexcept -> const NormalityProfile& {
    return impl_->active_profile;
}

auto CoresetEvolution::BindKnowledgeStore(
    std::shared_ptr<knowledge::KnowledgeStore> ks) noexcept -> void {
    impl_->knowledge_store = std::move(ks);
}

auto CoresetEvolution::FullRebuild(const std::filesystem::path& save_path) noexcept
    -> Result<void> {
    // Full rebuild: extract standby bank vectors + buffer candidates,
    // do full greedy coreset, persist to .bin + .profile.yaml.
    // Called from Stop() or explicitly by the caller.

    auto candidates = impl_->buffer.DrainAll();
    auto dim = impl_->active_profile.dim;

    if (dim == 0) {
        return {};
    }

    // Gather vectors: standby bank + buffer candidates
    std::vector<float> merged;
    if (impl_->standby_bank && impl_->standby_bank->NumSamples() > 0) {
        merged = impl_->standby_bank->ExtractAllVectors();
    }

    // Merge candidates
    for (auto& c : candidates) {
        auto n = c.grid_h * c.grid_w * c.dim;
        merged.insert(merged.end(), c.patch_vectors.get(), c.patch_vectors.get() + n);
    }

    if (merged.empty()) {
        return {};
    }

    auto total = merged.size() / dim;
    auto target = std::min(impl_->cfg.target_size, total);

    using namespace sai::embedding;
    EmbeddingMeta meta;
    meta.model_name = "full_rebuild";
    meta.type = EmbeddingType::Patch;
    meta.dim = dim;
    meta.count = total;
    meta.grid = {total, 1};
    auto emb = Embedding::FromCpu(std::move(merged), std::move(meta));
    std::vector<const Embedding*> ptrs{&emb};

    auto fb_result = FeatureBank::BuildWithGreedyCoreset(ptrs, dim, target);
    if (!fb_result.has_value()) {
        return tl::make_unexpected(fb_result.error());
    }

    auto new_bank = std::make_unique<FeatureBank>(std::move(*fb_result));

    // Compute new profile from the rebuilt bank (fast: sqrt(N) sampling,
    // avoids O(N²·D) exhaustive k-NN — typically < 1 s for 10k samples).
    auto sample_count = static_cast<std::size_t>(std::sqrt(
        static_cast<double>(new_bank->NumSamples())));
    auto new_profile = NormalityProfile::ComputeFast(
        *new_bank, impl_->cfg.normality_k, std::max(sample_count, std::size_t{1}));

    // Compute mean displacement for drift detection
    float mean_displacement = 0.0F;
    if (impl_->active_profile.num_samples > 0 && new_profile.num_samples > 0) {
        mean_displacement = std::abs(new_profile.mean - impl_->active_profile.mean);
    }

    // Save new coreset to file
    auto save_result = new_bank->SaveToFile(save_path);
    if (!save_result.has_value()) {
        return save_result;
    }

    // Save profile alongside
    auto profile_result = new_profile.SaveToYaml(
        fs::path(save_path).string() + ".profile.yaml");
    if (!profile_result.has_value()) {
        return profile_result;
    }

    // Backup old standby bank if configured
    if (impl_->cfg.backup_old_bank && impl_->standby_bank
        && impl_->standby_bank->NumSamples() > 0) {
        auto backup_path = save_path.string() + ".backup."
                         + std::to_string(
                             std::chrono::system_clock::now().time_since_epoch().count())
                         + ".bin";
        auto backup_result = impl_->standby_bank->SaveToFile(backup_path);
        if (!backup_result.has_value()) {
            sai::infra::Logger::Get("detection").Log(
                sai::infra::LogLevel::Warning,
                "CoresetEvolution: backup save failed: {}",
                backup_result.error().message);
        }
    }

    // Swap new bank into PatchCore; old active becomes new standby
    auto old = impl_->detector.SwapFeatureBank(std::move(new_bank));
    impl_->standby_bank = std::move(old);

    // Update profiles
    impl_->standby_profile = std::move(impl_->active_profile);
    impl_->active_profile = std::move(new_profile);

    // Remember save path for Stop()
    impl_->save_path = save_path;

    {
        std::lock_guard lock(impl_->stats_mutex);
        impl_->stats.update_count++;
    }

    // Record evolution event to KnowledgeStore
    if (impl_->knowledge_store) {
        sai::knowledge::KnowledgeRecord props;
        props.fields["event_type"] = sai::knowledge::FieldValue{
            std::string{"FullRebuild"}};
        props.fields["mean_displacement"] = sai::knowledge::FieldValue{
            static_cast<double>(mean_displacement)};
        props.fields["frames_added"] = sai::knowledge::FieldValue{
            static_cast<std::int64_t>(candidates.size())};

        auto node_result = impl_->knowledge_store->InsertNode(
            "CoresetEvolutionEvent", std::move(props));
        (void)node_result;
    }

    // Concept drift check
    impl_->CheckDrift(mean_displacement);

    return {};
}

}  // namespace sai::detection
