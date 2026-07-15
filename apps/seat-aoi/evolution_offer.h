// evolution_offer.h — Reusable CoresetEvolution AssessAndOffer helper
#pragma once

#include <sai/detection/coreset_evolution.h>
#include <sai/detection/patch_core.h>
#include <sai/reasoner/reasoner.h>

namespace seat_aoi {

/// Offer a detection context to a CoresetEvolution instance.
/// No-op if evolution is not running or knn_distances are empty.
/// Extracted because the same 15-parameter AssessAndOffer call appears
/// in both AppBuilder (headless callback) and GuiRunner (GUI callback).
inline void OfferToEvolution(
    sai::detection::CoresetEvolution& evo,
    const sai::detection::PatchCore::DetectionContext& ctx,
    const sai::reasoner::ReasoningResult& result) noexcept {
    if (!evo.IsRunning()) return;
    if (ctx.knn_distances.empty()) return;
    evo.AssessAndOffer(
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

}  // namespace seat_aoi
