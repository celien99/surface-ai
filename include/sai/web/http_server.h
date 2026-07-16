// http_server.h — Embedded REST API server for headless web dashboard
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sai/reasoner/reasoner.h>
#include <sai/pipeline/pipeline.h>

// Forward-declare httplib types to avoid header leakage.
namespace httplib {
class Server;
}  // namespace httplib

namespace sai::web {

// Lightweight inspection result summary for the web API.
struct InspectionSummary {
    int frame_id = 0;
    std::string verdict;
    double severity = 0.0;
    std::string recommendation;
    std::vector<std::string> defects;
    std::string timestamp;
};

// HttpServer: embedded HTTP server for headless deployments.
// Serves JSON REST API endpoints and a static dashboard HTML page.
// Uses cpp-httplib (header-only, MIT license).
//
// Lifecycle:
//   HttpServer server(8080);
//   server.Start();
//   // ... pipeline runs, call Update* from main thread ...
//   server.Stop();

class HttpServer {
public:
    explicit HttpServer(int port = 8080);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    auto operator=(const HttpServer&) -> HttpServer& = delete;

    // Start serving in a background thread.
    void Start();
    // Stop the server and join the thread.
    void Stop();

    // Data providers — thread-safe, called from pipeline callback.
    void UpdateResult(const InspectionSummary& summary);
    void UpdateMetrics(const std::vector<sai::pipeline::StageMetrics>& metrics);

    // Set server start time for uptime reporting.
    void SetStartTime(std::chrono::steady_clock::time_point t);

private:
    // Metrics snapshot (copyable, no atomics).
    struct MetricsSnapshot {
        std::string stage_id;
        std::size_t frames_processed = 0;
        std::size_t frames_failed = 0;
        double avg_latency_us = 0.0;
        std::size_t queue_depth = 0;
    };

    int port_;
    std::shared_ptr<httplib::Server> svr_;
    std::jthread thread_;
    std::atomic<bool> running_{false};
    std::chrono::steady_clock::time_point start_time_;

    // Thread-safe storage for latest results
    std::mutex mutex_;
    std::vector<InspectionSummary> history_;
    static constexpr std::size_t kMaxHistory = 1000;
    std::vector<MetricsSnapshot> metrics_;
    int ok_count_ = 0, ng_count_ = 0, warn_count_ = 0, uncertain_count_ = 0;
};

}  // namespace sai::web
