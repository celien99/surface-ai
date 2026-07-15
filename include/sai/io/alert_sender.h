// alert_sender.h — NG/WARN alert notifications (webhook + email)
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai::io {

// AlertInfo: structured data for an inspection alert.
struct AlertInfo {
    std::string verdict;                         // NG / WARN / UNCERTAIN
    double severity = 0.0;
    std::string recommendation;
    std::vector<std::string> defect_labels;
    int frame_id = 0;
    std::string timestamp;
};

// IAlertSender: sends notifications when non-OK verdicts occur.
class IAlertSender {
public:
    virtual ~IAlertSender() = default;
    [[nodiscard]] virtual auto SendAlert(const AlertInfo& info) noexcept -> Result<void> = 0;
};

// WebhookAlertSender: posts JSON to a webhook URL.
// Supports DingTalk, WeChat Work, and generic webhook formats.
class WebhookAlertSender final : public IAlertSender {
public:
    explicit WebhookAlertSender(std::string webhook_url);
    ~WebhookAlertSender() override = default;

    [[nodiscard]] auto SendAlert(const AlertInfo& info) noexcept -> Result<void> override;

private:
    std::string webhook_url_;
};

// CompositeAlertSender: fans out alerts to multiple senders.
class CompositeAlertSender final : public IAlertSender {
public:
    void AddSender(std::shared_ptr<IAlertSender> sender);

    [[nodiscard]] auto SendAlert(const AlertInfo& info) noexcept -> Result<void> override;

private:
    std::vector<std::shared_ptr<IAlertSender>> senders_;
};

}  // namespace sai::io
