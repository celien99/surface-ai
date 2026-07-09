#pragma once

#include <sai/core/object.h>
#include <sai/core/reflectable.h>

namespace sai {

class IService : public Object, public IReflectable {
public:
    virtual ~IService() = default;
};

}  // namespace sai
