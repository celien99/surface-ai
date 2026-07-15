// alert_sender.cpp — Webhook alert notification implementation

#include <sai/io/alert_sender.h>

#include <chrono>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

// Use libcurl for HTTP POST (available on all platforms).
// For a lighter dependency, we use a simple socket-based POST.
// cpp-httplib client mode is also available (same lib as http_server).

// Minimal HTTP POST using POSIX sockets (no external dependency beyond libc).
// Falls back gracefully if network is unavailable.
#ifdef __linux__
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#endif

namespace sai::io {

// ── WebhookAlertSender ──────────────────────────────────────────────

WebhookAlertSender::WebhookAlertSender(std::string webhook_url)
    : webhook_url_(std::move(webhook_url)) {}

auto WebhookAlertSender::SendAlert(const AlertInfo& info) noexcept -> Result<void> {
    // Build markdown message content
    std::ostringstream md;
    md << "## 🏭 Surface AI Inspection Alert\n\n"
       << "> **Verdict:** " << info.verdict << "\n"
       << "> **Severity:** " << info.severity << "\n"
       << "> **Frame:** #" << info.frame_id << "\n"
       << "> **Time:** " << info.timestamp << "\n\n";

    if (!info.defect_labels.empty()) {
        md << "**Defects detected:**\n";
        for (auto& d : info.defect_labels) {
            md << "- " << d << "\n";
        }
        md << "\n";
    }

    if (!info.recommendation.empty()) {
        md << "**Recommendation:** " << info.recommendation << "\n";
    }

    // Build JSON payload (DingTalk / WeChat Work compatible format)
    nlohmann::json payload;
    payload["msgtype"] = "markdown";
    payload["markdown"]["title"] = "Surface AI: " + info.verdict;
    payload["markdown"]["text"] = md.str();

    std::string body = payload.dump();

#ifdef __linux__
    // Parse URL to get host and path
    std::string url = webhook_url_;
    // Strip https:// or http://
    std::string host, path;
    bool use_ssl = false;
    int default_port = 80;

    if (url.find("https://") == 0) {
        url = url.substr(8);
        use_ssl = true;
        default_port = 443;
    } else if (url.find("http://") == 0) {
        url = url.substr(7);
    }

    auto path_start = url.find('/');
    if (path_start != std::string::npos) {
        host = url.substr(0, path_start);
        path = url.substr(path_start);
    } else {
        host = url;
        path = "/";
    }

    // Resolve host
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(default_port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        return {};  // Network unavailable — silent failure
    }

    int sock = socket(res->ai_family, res->ai_socktype, 0);
    if (sock < 0) {
        freeaddrinfo(res);
        return {};  // Socket creation failed — silent failure
    }

    // Connect with timeout
    struct timeval tv{2, 0};  // 2 second timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sock, res->ai_addr, res->ai_addrlen) != 0) {
        close(sock);
        freeaddrinfo(res);
        return {};  // Connection failed — silent failure
    }
    freeaddrinfo(res);

    // Send HTTP POST
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;

    std::string req_str = req.str();
    ::send(sock, req_str.c_str(), req_str.size(), 0);

    // Read response (fire-and-forget — don't block the pipeline)
    char buf[1024];
    ::recv(sock, buf, sizeof(buf) - 1, 0);
    close(sock);
#endif

    return {};
}

// ── CompositeAlertSender ────────────────────────────────────────────

void CompositeAlertSender::AddSender(std::shared_ptr<IAlertSender> sender) {
    if (sender) senders_.push_back(std::move(sender));
}

auto CompositeAlertSender::SendAlert(const AlertInfo& info) noexcept -> Result<void> {
    for (auto& sender : senders_) {
        (void)sender->SendAlert(info);  // Best-effort, don't stop on failure
    }
    return {};
}

}  // namespace sai::io
