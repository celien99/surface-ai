#pragma once

#include <cstdint>
#include <string_view>

namespace sai {

using TypeId = std::uint64_t;

namespace detail {
constexpr TypeId Fnv1aHash(std::string_view name) noexcept {
    TypeId hash = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
    for (char ch : name) {
        hash ^= static_cast<TypeId>(static_cast<unsigned char>(ch));
        hash *= 1099511628211ULL;  // FNV-1a 64-bit prime
    }
    return hash;
}
}  // namespace detail

}  // namespace sai

#define SAI_DECLARE_TYPE_ID(QualifiedName)                                \
    static constexpr std::string_view kStaticTypeName = #QualifiedName;   \
    static constexpr ::sai::TypeId kStaticTypeId =                        \
        ::sai::detail::Fnv1aHash(kStaticTypeName);                        \
    [[nodiscard]] auto TypeId() const noexcept -> ::sai::TypeId override { \
        return kStaticTypeId;                                             \
    }
