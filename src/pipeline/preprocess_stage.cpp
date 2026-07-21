#include "stage_nodes.h"

#include <sai/image/preprocess.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

namespace sai::pipeline {

PreprocessStage::PreprocessStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    // Parse steps from YAML config. Example:
    //   config:
    //     steps: [debayer, white_balance, resize]
    //     resize_width: 1024
    //     resize_height: 1024
    std::vector<sai::image::PreprocessFn> steps;

    if (auto steps_node = config["steps"]; steps_node.IsSequence()) {
        for (auto step : steps_node) {
            auto name = step.as<std::string>("");
            if (name == "debayer") {
                steps.push_back(sai::image::MakeDebayer());
            } else if (name == "white_balance") {
                float r = config["wb_r"].as<float>(1.0f);
                float g = config["wb_g"].as<float>(1.0f);
                float b = config["wb_b"].as<float>(1.0f);
                steps.push_back(sai::image::MakeWhiteBalance(r, g, b));
            } else if (name == "flat_field") {
                // FlatField needs a correction frame — use identity passthrough
                // for now. Real correction frame comes from calibration data.
            } else if (name == "resize") {
                auto w = config["resize_width"].as<std::size_t>(1024);
                auto h = config["resize_height"].as<std::size_t>(1024);
                steps.push_back(sai::image::MakeResize(w, h));
            }
        }
    }

    chain_ = sai::image::Compose(std::move(steps));
    stub_ = false;
}

auto PreprocessStage::GetType() const noexcept -> StageType { return StageType::Preprocess; }
auto PreprocessStage::GetId() const -> std::string_view { return id_; }

auto PreprocessStage::OnInitialize(Context&) -> Result<void> {
    return {};
}

auto PreprocessStage::OnStart(Context&) -> Result<void> { return {}; }
auto PreprocessStage::OnStop(Context&) -> Result<void> { return {}; }

auto PreprocessStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        return StageOutput::MakeWithContext(input, std::move(*failure));
    }

    if (auto* img = input.GetIf<sai::image::RawImage>()) {
        const auto meta = img->Meta();
        if (!stub_ && chain_) {
            auto raw = std::make_unique<sai::image::RawImage>(std::move(*img));
            auto result = chain_(std::move(raw));
            if (result) {
                // Compose returns unique_ptr<Image>; SurfaceImage inherits from Image.
                // We need to produce a SurfaceImage variant. Since downstream
                // stages expect SurfaceImage, we assume the chain produces one.
                auto* surf = dynamic_cast<sai::image::SurfaceImage*>(result->get());
                if (surf) {
                    result->release();  // transfer ownership
                    return StageOutput::MakeWithContext(input, std::move(*surf));
                }
                // Fallback: wrap buffer as SurfaceImage
                auto meta = (*result)->Meta();
                const auto* data = (*result)->Data();
                auto size = (*result)->SizeBytes();
                std::vector<std::uint8_t> buffer(data, data + size);
                return StageOutput::MakeWithContext(input,
                    sai::image::SurfaceImage::FromOwnedBuffer(std::move(buffer), meta));
            }
            if (!result) {
                return StageOutput::MakeWithContext(input, PipelineFailure{
                    .code = result.error().code,
                    .stage_id = id_,
                    .message = result.error().message,
                    .surface_id = meta.surface_id,
                    .position_id = meta.position_id,
                });
            }
        }
        // Stub / fallback: wrap RawImage buffer as SurfaceImage
        const auto* data = img->Data();
        auto size = img->SizeBytes();
        std::vector<std::uint8_t> buffer(data, data + size);
        return StageOutput::MakeWithContext(input,
            sai::image::SurfaceImage::FromOwnedBuffer(std::move(buffer), meta));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Preprocess expects RawImage input"});
}

}  // namespace sai::pipeline
