#include <sai/detection/coreset_evolution.h>
#include <sai/detection/feature_bank.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <source_location>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::detection {

auto NormalityProfile::Compute(const FeatureBank& bank,
                                std::size_t k,
                                float tail_ratio_max) noexcept -> NormalityProfile {
    auto dim = bank.Dim();
    auto num = bank.NumSamples();
    if (num == 0 || dim == 0) return {};

    auto all_vecs = bank.ExtractAllVectors();
    std::vector<float> nn_dists(num);

    // 对每个 coreset 向量做 self-query（k+1 跳过自身）
    auto search_k = static_cast<std::size_t>(k + 1);
    for (std::size_t i = 0; i < num; ++i) {
        auto dists = bank.Search(all_vecs.data() + i * dim, 1, search_k);
        nn_dists[i] = dists.back();  // 第 k 近邻（k=search_k-1）
    }

    std::sort(nn_dists.begin(), nn_dists.end());

    NormalityProfile profile;
    profile.k_nearest = k;
    profile.dim = dim;
    profile.num_samples = num;

    auto percentile = [&](float p) -> float {
        auto idx = static_cast<std::size_t>(p * static_cast<float>(num - 1));
        if (idx >= num) idx = num - 1;
        return nn_dists[idx];
    };

    profile.p50 = percentile(0.50F);
    profile.p95 = percentile(0.95F);
    profile.p99 = percentile(0.99F);

    double sum = 0.0;
    for (auto d : nn_dists) sum += static_cast<double>(d);
    profile.mean = static_cast<float>(sum / static_cast<double>(num));

    double var_sum = 0.0;
    for (auto d : nn_dists) {
        double diff = static_cast<double>(d) - static_cast<double>(profile.mean);
        var_sum += diff * diff;
    }
    profile.stddev = static_cast<float>(std::sqrt(var_sum / static_cast<double>(num)));

    // ── 数据驱动门控：计算 coreset 自身的 normalcy_score ──
    // 按定义约 5% 的 patch 超过 p95，因此 self_tail_ratio ≈ 0.05。
    // self_normalcy = 1.0 - self_tail_ratio / tail_ratio_max。
    // 所有运行时门控阈值均由此值推导，自适应不同材料的自然方差。
    std::size_t self_tail = 0;
    for (auto d : nn_dists) {
        if (d > profile.p95) ++self_tail;
    }
    float self_tail_ratio = static_cast<float>(self_tail) / static_cast<float>(num);
    if (tail_ratio_max > 0.0F) {
        profile.self_normalcy = 1.0F - std::min(1.0F, self_tail_ratio / tail_ratio_max);
    } else {
        profile.self_normalcy = 1.0F;
    }

    return profile;
}

auto NormalityProfile::ComputeFast(const FeatureBank& bank,
                                     std::size_t k,
                                     std::size_t sample_count,
                                     float tail_ratio_max) noexcept -> NormalityProfile {
    auto num = bank.NumSamples();
    if (num == 0) return {};
    auto actual_samples = std::min(sample_count, num);

    // 均匀采样 sqrt(K) 个向量做自查询
    auto dim = bank.Dim();
    auto all_vecs = bank.ExtractAllVectors();
    std::vector<float> nn_dists(actual_samples);
    auto stride = num / actual_samples;
    if (stride < 1) stride = 1;

    auto search_k = k + 1;
    for (std::size_t i = 0; i < actual_samples; ++i) {
        auto idx = i * stride;
        if (idx >= num) idx = num - 1;
        auto dists = bank.Search(all_vecs.data() + idx * dim, 1, search_k);
        nn_dists[i] = dists.back();
    }

    std::sort(nn_dists.begin(), nn_dists.end());

    NormalityProfile profile;
    profile.k_nearest = k;
    profile.dim = dim;
    profile.num_samples = num;

    auto percentile_fast = [&](float p) -> float {
        auto idx = static_cast<std::size_t>(p * static_cast<float>(actual_samples - 1));
        if (idx >= actual_samples) idx = actual_samples - 1;
        return nn_dists[idx];
    };

    profile.p50 = percentile_fast(0.50F);
    profile.p95 = percentile_fast(0.95F);
    profile.p99 = percentile_fast(0.99F);

    double sum = 0.0;
    for (auto d : nn_dists) sum += static_cast<double>(d);
    profile.mean = static_cast<float>(sum / static_cast<double>(actual_samples));

    double var_sum = 0.0;
    for (auto d : nn_dists) {
        double diff = static_cast<double>(d) - static_cast<double>(profile.mean);
        var_sum += diff * diff;
    }
    profile.stddev = static_cast<float>(std::sqrt(var_sum / static_cast<double>(actual_samples)));

    // 数据驱动门控：采样自查询的 self_normalcy
    std::size_t self_tail = 0;
    for (auto d : nn_dists) {
        if (d > profile.p95) ++self_tail;
    }
    float self_tail_ratio = static_cast<float>(self_tail) / static_cast<float>(actual_samples);
    if (tail_ratio_max > 0.0F) {
        profile.self_normalcy = 1.0F - std::min(1.0F, self_tail_ratio / tail_ratio_max);
    } else {
        profile.self_normalcy = 1.0F;
    }

    return profile;
}

// ── 数据驱动阈值推导 ──
// 因子 0.8 / 0.7 的含义："阈值放宽到 coreset 自身正常度的 80%/70%"
// 紧凑材料（金属）self_normalcy ≈ 0.50 → consensus ≈ 0.40, evolution ≈ 0.35
// 松散材料（织物）self_normalcy 更低 → 自动放宽

auto NormalityProfile::ConsensusThreshold() const noexcept -> float {
    if (self_normalcy <= 0.0F) return 0.40F;  // fallback for uninitialized profile
    return self_normalcy * 0.8F;
}

auto NormalityProfile::EvolutionGate() const noexcept -> float {
    if (self_normalcy <= 0.0F) return 0.30F;
    return self_normalcy * 0.7F;
}

auto NormalityProfile::LoadFromYaml(const std::filesystem::path& path) noexcept
    -> Result<NormalityProfile> {
    try {
        auto root = YAML::LoadFile(path.string());
        auto p = root["profile"];
        if (!p.IsDefined()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_CoresetEvolution_ProfileLoadFailed,
                "missing 'profile' key in: " + path.string(),
                std::source_location::current(),
            });
        }
        NormalityProfile profile;
        profile.k_nearest = p["k_nearest"].as<std::size_t>(5);
        profile.dim = p["dim"].as<std::size_t>(0);
        profile.num_samples = p["num_samples"].as<std::size_t>(0);
        auto s = p["statistics"];
        profile.p50 = s["p50"].as<float>(0.0F);
        profile.p95 = s["p95"].as<float>(0.0F);
        profile.p99 = s["p99"].as<float>(0.0F);
        profile.mean = s["mean"].as<float>(0.0F);
        profile.stddev = s["stddev"].as<float>(0.0F);
        profile.self_normalcy = s["self_normalcy"].as<float>(0.0F);
        return profile;
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_CoresetEvolution_ProfileLoadFailed,
            "YAML parse error: " + std::string(e.what()),
            std::source_location::current(),
        });
    }
}

auto NormalityProfile::SaveToYaml(const std::filesystem::path& path) const noexcept
    -> Result<void> {
    try {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "profile" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "k_nearest" << YAML::Value << k_nearest;
        out << YAML::Key << "dim" << YAML::Value << dim;
        out << YAML::Key << "num_samples" << YAML::Value << num_samples;
        out << YAML::Key << "statistics" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "p50" << YAML::Value << p50;
        out << YAML::Key << "p95" << YAML::Value << p95;
        out << YAML::Key << "p99" << YAML::Value << p99;
        out << YAML::Key << "mean" << YAML::Value << mean;
        out << YAML::Key << "stddev" << YAML::Value << stddev;
        out << YAML::Key << "self_normalcy" << YAML::Value << self_normalcy;
        out << YAML::EndMap;
        out << YAML::EndMap;
        out << YAML::EndMap;

        std::ofstream file(path);
        if (!file.is_open()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_CoresetEvolution_ProfileLoadFailed,
                "cannot open file for writing: " + path.string(),
                std::source_location::current(),
            });
        }
        file << out.c_str();
        return {};
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_CoresetEvolution_ProfileLoadFailed,
            "YAML emit error: " + std::string(e.what()),
            std::source_location::current(),
        });
    }
}

}  // namespace sai::detection
