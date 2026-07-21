// embedder_factory.h — Shared DINOv2 PatchEmbedder factory
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

/// Create a TensorRT + DINOv2 PatchEmbedder.
/// Shared by app_builder.cpp (app assembly) and coreset_builder.cpp (standalone coreset build).
/// Returns the embedder shared_ptr, or an error if any creation step fails.
inline auto CreateDinoV2PatchEmbedder(
    const std::filesystem::path& engine_path,
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
            "DINOv2 engine not found: " + engine_path.string()});
    }

    auto engine = std::make_shared<TensorRtEngine>(/*device_ordinal=*/0);
    std::cout << "Inference: TensorRtEngine (GPU)\n";

    DinoV3Config cfg;
    cfg.engine_path = engine_path;
    cfg.image_size = image_size;
    cfg.patch_size = patch_size;
    cfg.embed_dim = embed_dim;

    // Load the TensorRT engine file before creating the adapter.
    // The engine must be deserialized first so OutputBindings() returns
    // the model's I/O tensors.
    // DINOv2 ViT engine has 2 bindings:
    //   input:  "pixel_values"      (1, 3, H, W)  — set at inference time
    //   output: "last_hidden_state" (1, grid²+1, embed_dim)  — includes CLS token
    auto grid_size = image_size / patch_size;
    std::size_t output_floats = (grid_size * grid_size + 1) * embed_dim;  // patches + CLS token
    long is = static_cast<long>(image_size);
    long gs = static_cast<long>(grid_size);
    long ed = static_cast<long>(embed_dim);
    std::vector<sai::inference::TensorBinding> inputs = {
        {"pixel_values", {1, 3, is, is},
         static_cast<std::size_t>(is * is * 3), nullptr}
    };
    std::vector<sai::inference::TensorBinding> outputs = {
        {"last_hidden_state", {1, gs * gs + 1, ed},
         output_floats * sizeof(float), nullptr}
    };
    auto load_result = engine->Load(engine_path, std::move(inputs), std::move(outputs));
    if (!load_result) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineLoadFailed,
            "DINOv2 engine load failed: " + load_result.error().message});
    }

    auto adapter = DinoV3Adapter::Create(*engine, cfg);
    if (!adapter) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            "DinoAdapter creation failed: " + adapter.error().message});
    }
    auto embedder = PatchEmbedder::Create(std::move(*adapter));
    if (!embedder) {
        return tl::make_unexpected(sai::ErrorInfo{
            sai::ErrorCode::Inference_EngineExecutionFailed,
            "PatchEmbedder creation failed"});
    }

    auto result = std::make_shared<PatchEmbedder>(std::move(*embedder));
    result->SetEngineHolder(engine);  // keep engine alive ≥ embedder lifetime
    std::cout << "Embedder: PatchEmbedder (DINOv2, dim=" << embed_dim << ")\n";
    return result;
}

}  // namespace seat_aoi
