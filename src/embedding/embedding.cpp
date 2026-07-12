// embedding.cpp — 批次 3.2 Embedding 双存储数据类型实现
#include <sai/embedding/embedding.h>

namespace sai::embedding {

Embedding Embedding::FromGpu(sai::memory::PooledPtr<std::uint8_t> device_data,
                              EmbeddingMeta meta) noexcept {
    Embedding emb;
    emb.device_buffer_ = std::move(device_data);
    emb.meta_ = std::move(meta);
    emb.on_gpu_ = true;
    return emb;
}

Embedding Embedding::FromCpu(std::vector<float> data, EmbeddingMeta meta) noexcept {
    Embedding emb;
    emb.cpu_data_ = std::move(data);
    emb.meta_ = std::move(meta);
    emb.on_gpu_ = false;
    return emb;
}

auto Embedding::Data() const noexcept -> const float* {
    if (on_gpu_) {
        return reinterpret_cast<const float*>(device_buffer_.Get());
    }
    return cpu_data_.data();
}

auto Embedding::SizeBytes() const noexcept -> std::size_t {
    return meta_.count * meta_.dim * sizeof(float);
}

Embedding::~Embedding() noexcept = default;

}  // namespace sai::embedding
