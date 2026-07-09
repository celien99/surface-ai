#pragma once

#include <sai/core/type_id.h>

namespace sai {

class IReflectable {
public:
    virtual ~IReflectable() = default;

    [[nodiscard]] virtual auto TypeId() const noexcept -> sai::TypeId = 0;
};

}  // namespace sai
