#include <sai/plugin/license_manager.h>

#include <source_location>

namespace sai {

auto LicenseManager::Validate(std::string_view license_token) const -> Result<void> {
    if (license_token.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Plugin_LicenseInvalid,
            "license token is empty",
            std::source_location::current(),
        });
    }
    return {};
}

}  // namespace sai
