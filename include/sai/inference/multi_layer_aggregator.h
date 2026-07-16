// multi_layer_aggregator.h — DINOv2/v3 多层特征聚合（Concat / Mean / Group）
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sai::inference {

// 聚合策略
enum class AggMethod : std::uint8_t {
    Concat,  // 通道维度拼接：D × N_layers
    Mean,    // 逐元素平均：D（维度不变）
    Group,   // 组内平均 + 跨组拼接：D × N_groups
};

// 多层聚合配置
struct MultiLayerConfig {
    std::vector<int> layers;                // 层索引列表（支持负索引，如 {-2,-3,-4}）
    AggMethod method = AggMethod::Mean;     // 聚合策略
    std::vector<std::vector<int>> groups;   // Group 模式下的分组：
                                            //   {{-12,-13}, {-14,-15}, {-16,-17,-18}}
    std::size_t embed_dim = 1024;           // 单层特征维度
};

// MultiLayerAggregator——将多个 Transformer 层的 patch 特征聚合为单层特征。
//
// 设计决策：
// - DINOv2/v3 的浅层编码纹理信息，深层编码语义信息。实验表明
//   concat/group 多层特征能提升异常检测精度（尤其是小缺陷）。
// - Mean 保持维度不变（D），适合与现有 Pipeline 无缝集成。
// - Concat 产生 D × N 维特征，Group 产生 D × G 维（G = 组数）。
// - 输入 layer_features[N_layers]：每个元素是 count × D 的扁平 float 数组。
// - 输出：count × D' 的扁平数组，其中 D' = D (Mean) | D×N (Concat) | D×G (Group)。
class MultiLayerAggregator final {
public:
    explicit MultiLayerAggregator(MultiLayerConfig cfg) noexcept;

    // 聚合多层特征
    // layer_features: 每个元素对应一个层的扁平特征（count × D）
    // count: 每个层的特征向量数（所有层必须相同）
    [[nodiscard]] auto Aggregate(const std::vector<const float*>& layer_features,
                                  std::size_t count) const noexcept -> std::vector<float>;

    [[nodiscard]] auto OutputDim() const noexcept -> std::size_t { return output_dim_; }
    [[nodiscard]] auto Method() const noexcept -> AggMethod { return cfg_.method; }

private:
    MultiLayerConfig cfg_;
    std::size_t output_dim_;
    std::unordered_map<int, std::size_t> layer_index_;  // layer_idx -> index in layers vector
};

}  // namespace sai::inference
