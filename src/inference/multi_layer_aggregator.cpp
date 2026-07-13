// multi_layer_aggregator.cpp — 多层特征聚合实现
#include <algorithm>
#include <cstddef>
#include <vector>

#include <sai/inference/multi_layer_aggregator.h>

namespace sai::inference {

MultiLayerAggregator::MultiLayerAggregator(MultiLayerConfig cfg) noexcept
    : cfg_(std::move(cfg)) {
    switch (cfg_.method) {
        case AggMethod::Concat:
            output_dim_ = cfg_.embed_dim * cfg_.layers.size();
            break;
        case AggMethod::Mean:
            output_dim_ = cfg_.embed_dim;
            break;
        case AggMethod::Group:
            output_dim_ = cfg_.embed_dim * cfg_.groups.size();
            break;
    }
}

auto MultiLayerAggregator::Aggregate(
    const std::vector<const float*>& layer_features,
    std::size_t count) const noexcept -> std::vector<float> {
    if (layer_features.empty() || count == 0) return {};

    auto n_layers = layer_features.size();
    auto D = cfg_.embed_dim;

    if (cfg_.method == AggMethod::Mean) {
        std::vector<float> out(count * D, 0.0f);
        float inv_n = 1.0f / static_cast<float>(n_layers);
        for (std::size_t l = 0; l < n_layers; ++l) {
            for (std::size_t i = 0; i < count * D; ++i) {
                out[i] += layer_features[l][i] * inv_n;
            }
        }
        return out;
    }

    if (cfg_.method == AggMethod::Concat) {
        std::vector<float> out(count * output_dim_);
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t offset = 0;
            for (std::size_t l = 0; l < n_layers; ++l) {
                std::copy(layer_features[l] + i * D,
                          layer_features[l] + (i + 1) * D,
                          out.data() + i * output_dim_ + offset);
                offset += D;
            }
        }
        return out;
    }

    // Group 方法
    std::vector<float> out(count * output_dim_, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t out_offset = 0;
        for (const auto& group : cfg_.groups) {
            // 组内平均
            std::vector<float> group_mean(D, 0.0f);
            float inv_g = 1.0f / static_cast<float>(group.size());
            for (int layer_idx : group) {
                // 找到 layer_idx 在 layer_features 中的位置
                auto it = std::find(cfg_.layers.begin(), cfg_.layers.end(), layer_idx);
                if (it == cfg_.layers.end()) continue;
                auto l = static_cast<std::size_t>(std::distance(cfg_.layers.begin(), it));
                for (std::size_t j = 0; j < D; ++j) {
                    group_mean[j] += layer_features[l][i * D + j] * inv_g;
                }
            }
            std::copy(group_mean.begin(), group_mean.end(),
                      out.data() + i * output_dim_ + out_offset);
            out_offset += D;
        }
    }
    return out;
}

}  // namespace sai::inference
