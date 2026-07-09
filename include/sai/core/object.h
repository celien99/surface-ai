#pragma once

namespace sai {

class Object {
public:
    virtual ~Object() = default;

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = delete;
    Object& operator=(Object&&) = delete;

protected:
    Object() = default;
};

}  // namespace sai
