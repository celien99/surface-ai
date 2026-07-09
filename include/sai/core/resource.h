#pragma once

namespace sai {

class Resource {
public:
    Resource() noexcept = default;
    virtual ~Resource() noexcept = default;

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) noexcept = default;
    Resource& operator=(Resource&&) noexcept = default;

    [[nodiscard]] virtual bool IsValid() const noexcept = 0;
    virtual void Release() noexcept = 0;
};

}  // namespace sai
