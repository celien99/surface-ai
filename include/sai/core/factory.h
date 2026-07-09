#pragma once

#include <memory>

#include <sai/core/error.h>

namespace sai {

template <typename TInterface>
class Factory {
public:
    virtual ~Factory() = default;

    [[nodiscard]] virtual auto Create() -> Result<std::unique_ptr<TInterface>> = 0;
};

}  // namespace sai
