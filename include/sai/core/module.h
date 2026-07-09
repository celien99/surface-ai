#pragma once

#include <sai/core/error.h>
#include <sai/core/object.h>

namespace sai {

class Context;

class IModule : public Object {
public:
    virtual ~IModule() = default;

    virtual auto OnInitialize(Context& context) -> Result<void> = 0;
    virtual auto OnStart(Context& context) -> Result<void> = 0;
    virtual auto OnStop(Context& context) -> Result<void> = 0;
};

}  // namespace sai
