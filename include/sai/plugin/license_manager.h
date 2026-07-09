#pragma once

#include <string_view>

#include <sai/core/error.h>

namespace sai {

// Only the call contract is locked here, not a real validation algorithm —
// signature verification/online activation are deployment-specific
// implementation details (see §2 Responsibilities). Failure returns
// Plugin_LicenseInvalid, never throws (see §3 Design).
//
// Chosen rule for this batch: a non-empty token is valid, an empty token is
// invalid. This is deterministic and testable (both the valid and invalid
// paths are exercised), and keeps the "invalid" path meaningful rather than
// a stub that always succeeds; a real deployment plugs in real crypto later
// (see §2/§12) without changing this call contract.
class LicenseManager {
public:
    [[nodiscard]] auto Validate(std::string_view license_token) const -> Result<void>;
};

}  // namespace sai
