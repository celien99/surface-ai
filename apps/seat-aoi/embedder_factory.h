// embedder_factory.h — Shared DINOv3 PatchEmbedder factory
#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <sai/core/error.h>
#include <sai/embedding/embedder.h>
#include <sai/inference/dino_v3_adapter.h>
#include <sai/inference/tensorrt_engine.h>

namespace seat_aoi {

/// Create a TensorRT + DINOv3 PatchEmbedder.
/// Shared by app_builder.cpp (app assembly) and coreset_builder.cpp (standalone coreset build).
/// Returns the embedder shared_ptr, or an error if any creation step fails.
inline auto CreateDinoV3PatchEmbedder(
    std::string_view engine_path,
    std::size_t image_size,
    std::size_t patch_size,
    std::size_t embed_dim) -> sai::Result<std::shared_ptr<sai::embedding::PatchEmbedder>> {

    using sai::inference::TensorRtEngine;
    using sai::inference::DinoV3Config;
    using sai::inference::DinoV3Adapter;
    using sai::embedding::PatchEmbedder;

    if (!std::filesystem::exists(engine_path)) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Core_ConstructionFailed,
            "DINOv3 engine not found: " + std::string(engine_path)});
    }

    auto engine = std::make_shared<TensorRtEngine>(/*device_ordinal=*/0);
    std::cout << "Inference: TensorRtEngine (GPU)\n";

    DinoV3Config cfg;
    cfg.engine_path = engine_path;
    cfg.image_size = image_size;
    cfg.patch_size = patch_size;
    cfg.embed_dim = embed_dim;

    auto adapter = DinoV3Adapter::Create(*engine, cfg);
    if (!adapter) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            "DinoV3Adapter creation failed: " + adapter.error().message});
    }
    auto embedder = PatchEmbedder::Create(std::move(*adapter));
    if (!embedder) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder creation failed"});
    }

    auto result = std::make_shared<PatchEmbedder>(std::move(*embedder));
    std::cout << "Embedder: PatchEmbedder (DINOv3, dim=" << embed_dim << ")\n";
    return result;
}

}  // namespace seat_aoi
