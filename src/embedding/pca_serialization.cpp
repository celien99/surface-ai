// dimension_reducer.cpp — 批次 3.2 DimensionReducer PCA/Whitening 降维、评分、序列化与 Pooling 实现
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <sai/embedding/dimension_reducer.h>

namespace sai::embedding {

// ============================================================================
// Static SavePcaParams / LoadPcaParams — 序列化
// ============================================================================

auto DimensionReducer::SavePcaParams(const PcaParams& params,
                                      const std::filesystem::path& path) noexcept -> Result<void> {
    auto D = static_cast<std::uint64_t>(params.mean.size());
    auto k = static_cast<std::uint64_t>(params.target_dim);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "DimensionReducer::SavePcaParams: cannot open file: " + path.string(),
            std::source_location::current()});
    }

    // [D:u64][k:u64]
    file.write(reinterpret_cast<const char*>(&D), sizeof(D));
    file.write(reinterpret_cast<const char*>(&k), sizeof(k));

    // [mu: D×f32]
    file.write(reinterpret_cast<const char*>(params.mean.data()),
               static_cast<std::streamsize>(D * sizeof(float)));

    // [eigvals: k×f32]
    file.write(reinterpret_cast<const char*>(params.eigvals.data()),
               static_cast<std::streamsize>(k * sizeof(float)));

    // [components: k×D×f32]
    file.write(reinterpret_cast<const char*>(params.components.data()),
               static_cast<std::streamsize>(k * D * sizeof(float)));

    if (file.fail()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "DimensionReducer::SavePcaParams: write failed: " + path.string(),
            std::source_location::current()});
    }

    return {};
}

auto DimensionReducer::LoadPcaParams(const std::filesystem::path& path) noexcept -> Result<PcaParams> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "DimensionReducer::LoadPcaParams: cannot open file: " + path.string(),
            std::source_location::current()});
    }

    auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::uint64_t D = 0;
    std::uint64_t k = 0;
    file.read(reinterpret_cast<char*>(&D), sizeof(D));
    file.read(reinterpret_cast<char*>(&k), sizeof(k));

    auto expected_size = sizeof(std::uint64_t) * 2
        + D * sizeof(float)           // mu
        + k * sizeof(float)           // eigvals
        + k * D * sizeof(float);      // components

    if (file_size < expected_size || D == 0 || k == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "DimensionReducer::LoadPcaParams: invalid file format or dimensions: " + path.string(),
            std::source_location::current()});
    }

    PcaParams params;
    params.target_dim = static_cast<std::size_t>(k);
    params.mean.resize(static_cast<std::size_t>(D));
    params.eigvals.resize(static_cast<std::size_t>(k));
    params.components.resize(static_cast<std::size_t>(k * D));

    file.read(reinterpret_cast<char*>(params.mean.data()),
              static_cast<std::streamsize>(D * sizeof(float)));
    file.read(reinterpret_cast<char*>(params.eigvals.data()),
              static_cast<std::streamsize>(k * sizeof(float)));
    file.read(reinterpret_cast<char*>(params.components.data()),
              static_cast<std::streamsize>(k * D * sizeof(float)));

    if (file.fail()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "DimensionReducer::LoadPcaParams: read failed: " + path.string(),
            std::source_location::current()});
    }

    return params;
}


}  // namespace sai::embedding
