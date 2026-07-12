// feature_bank.cpp — FeatureBank 可移植 CPU 实现（FAISS IndexFlatL2）
#include <sai/detection/feature_bank.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <source_location>

#include <faiss/IndexFlat.h>

namespace sai::detection {

auto FeatureBank::LoadFromFile(const std::filesystem::path& path,
                               std::size_t dim) noexcept -> Result<FeatureBank> {
    // 读取文件
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "cannot open feature bank file: " + path.string(),
            std::source_location::current(),
        });
    }

    auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    auto element_size = sizeof(float);
    if (file_size % (dim * element_size) != 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "file size " + std::to_string(file_size)
                + " is not a multiple of dim " + std::to_string(dim),
            std::source_location::current(),
        });
    }

    auto num_samples = file_size / (dim * element_size);
    std::vector<float> buffer(num_samples * dim);
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size() * element_size));
    if (file.fail()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_FeatureBankLoadFailed,
            "failed to read feature bank file: " + path.string(),
            std::source_location::current(),
        });
    }

    // 创建 FAISS IndexFlatL2
    auto index = std::make_unique<faiss::IndexFlatL2>(static_cast<faiss::idx_t>(dim));
    index->add(static_cast<faiss::idx_t>(num_samples), buffer.data());

    FeatureBank bank;
    bank.index_ = std::move(index);
    bank.dim_ = dim;
    bank.num_samples_ = num_samples;
    return bank;
}

auto FeatureBank::Search(const float* query, std::size_t query_count,
                         std::size_t k) const noexcept -> std::vector<float> {
    if (query_count == 0 || k == 0 || !index_) {
        return {};
    }

    auto nq = static_cast<faiss::idx_t>(query_count);
    auto nk = static_cast<faiss::idx_t>(k);

    std::vector<float> distances(static_cast<std::size_t>(nq * nk));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(nq * nk));

    index_->search(nq, query, nk, distances.data(), labels.data());

    return distances;
}

FeatureBank::FeatureBank(FeatureBank&&) noexcept = default;
auto FeatureBank::operator=(FeatureBank&&) noexcept -> FeatureBank& = default;

FeatureBank::~FeatureBank() = default;

}  // namespace sai::detection
