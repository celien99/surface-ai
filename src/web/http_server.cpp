// http_server.cpp — Embedded HTTP server with REST API + dashboard

#include <sai/web/http_server.h>

#include <chrono>
#include <mutex>
#include <sstream>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace sai::web {

HttpServer::HttpServer(int port) : port_(port) {
    start_time_ = std::chrono::steady_clock::now();
}

HttpServer::~HttpServer() {
    Stop();
}

void HttpServer::Start() {
    if (running_.exchange(true)) return;

    svr_ = std::make_shared<httplib::Server>();

    thread_ = std::jthread([this](std::stop_token /*st*/) {
        // ── CORS headers ──────────────────────────────────────────
        svr_->set_default_headers({
            {"Access-Control-Allow-Origin", "*"},
            {"Access-Control-Allow-Methods", "GET, OPTIONS"},
            {"Content-Type", "application/json"},
        });

        // ── GET /api/status ───────────────────────────────────────
        svr_->Get("/api/status", [this](const httplib::Request&, httplib::Response& res) {
            auto now = std::chrono::steady_clock::now();
            auto uptime = std::chrono::duration<double>(now - start_time_).count();

            nlohmann::json j;
            {
                std::lock_guard<std::mutex> lk(mutex_);
                j["ok"] = ok_count_;
                j["ng"] = ng_count_;
                j["warn"] = warn_count_;
                j["uncertain"] = uncertain_count_;
                j["total"] = ok_count_ + ng_count_ + warn_count_ + uncertain_count_;
            }
            j["uptime_seconds"] = uptime;
            j["pipeline"] = "running";

            res.set_content(j.dump(2), "application/json");
        });

        // ── GET /api/metrics ──────────────────────────────────────
        svr_->Get("/api/metrics", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j = nlohmann::json::array();
            {
                std::lock_guard<std::mutex> lk(mutex_);
                for (auto& m : metrics_) {
                    nlohmann::json stage;
                    stage["id"] = m.stage_id;
                    stage["id"] = m.stage_id;
                    stage["frames_processed"] = m.frames_processed;
                    stage["frames_failed"] = m.frames_failed;
                    stage["avg_latency_us"] = m.avg_latency_us.load();
                    stage["queue_depth"] = m.queue_depth;
                    j.push_back(stage);
                }
            }
            res.set_content(j.dump(2), "application/json");
        });

        // ── GET /api/results/latest ───────────────────────────────
        svr_->Get("/api/results/latest", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j;
            {
                std::lock_guard<std::mutex> lk(mutex_);
                if (!history_.empty()) {
                    auto& r = history_.back();
                    j["frame_id"] = r.frame_id;
                    j["verdict"] = r.verdict;
                    j["severity"] = r.severity;
                    j["recommendation"] = r.recommendation;
                    j["defects"] = r.defects;
                    j["timestamp"] = r.timestamp;
                } else {
                    j = nullptr;
                }
            }
            res.set_content(j.dump(2), "application/json");
        });

        // ── GET /api/results/history ──────────────────────────────
        svr_->Get("/api/results/history", [this](const httplib::Request& req, httplib::Response& res) {
            int limit = 50;
            if (req.has_param("limit")) {
                limit = std::stoi(req.get_param_value("limit"));
            }
            nlohmann::json j = nlohmann::json::array();
            {
                std::lock_guard<std::mutex> lk(mutex_);
                auto start = history_.size() > static_cast<std::size_t>(limit)
                    ? history_.size() - static_cast<std::size_t>(limit) : 0;
                for (auto i = start; i < history_.size(); ++i) {
                    auto& r = history_[i];
                    nlohmann::json item;
                    item["frame_id"] = r.frame_id;
                    item["verdict"] = r.verdict;
                    item["severity"] = r.severity;
                    item["defects"] = r.defects;
                    j.push_back(item);
                }
            }
            res.set_content(j.dump(2), "application/json");
        });

        // ── GET /api/alerts ───────────────────────────────────────
        svr_->Get("/api/alerts", [this](const httplib::Request&, httplib::Response& res) {
            nlohmann::json j = nlohmann::json::array();
            {
                std::lock_guard<std::mutex> lk(mutex_);
                for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
                    if (it->verdict != "OK") {
                        nlohmann::json item;
                        item["frame_id"] = it->frame_id;
                        item["verdict"] = it->verdict;
                        item["severity"] = it->severity;
                        item["defects"] = it->defects;
                        j.push_back(item);
                        if (j.size() >= 20) break;  // last 20 alerts
                    }
                }
            }
            res.set_content(j.dump(2), "application/json");
        });

        // ── GET / → dashboard HTML ────────────────────────────────
        svr_->Get("/", [](const httplib::Request&, httplib::Response& res) {
            // Inline minimal dashboard HTML (also served from resources/web/ if available)
            std::string html = R"(<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surface AI Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}
h1{color:#58a6ff;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-align:center}
.card .value{font-size:32px;font-weight:bold}
.card .label{font-size:12px;color:#8b949e;margin-top:4px}
.ok .value{color:#3fb950}
.ng .value{color:#f85149}
.warn .value{color:#d2991d}
.uncertain .value{color:#a371f7}
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-size:12px;text-transform:uppercase}
tr:hover{background:#1c2128}
#alerts{max-height:300px;overflow-y:auto}
.badge{padding:2px 8px;border-radius:12px;font-size:11px;font-weight:bold}
.badge-ok{background:#3fb95022;color:#3fb950}
.badge-ng{background:#f8514922;color:#f85149}
.badge-warn{background:#d2991d22;color:#d2991d}
</style></head>
<body>
<h1>🏭 Surface AI — Inspection Dashboard</h1>
<div class="cards" id="cards">
<div class="card ok"><div class="value" id="cnt-ok">-</div><div class="label">OK</div></div>
<div class="card ng"><div class="value" id="cnt-ng">-</div><div class="label">NG</div></div>
<div class="card warn"><div class="value" id="cnt-warn">-</div><div class="label">WARN</div></div>
<div class="card uncertain"><div class="value" id="cnt-uncertain">-</div><div class="label">UNCERTAIN</div></div>
<div class="card"><div class="value" id="uptime">-</div><div class="label">Uptime</div></div>
</div>
<h3>Recent Alerts</h3>
<div id="alerts"><table><thead><tr><th>Frame</th><th>Verdict</th><th>Severity</th><th>Defects</th></tr></thead><tbody id="alert-rows"></tbody></table></div>
<h3 style="margin-top:20px">Pipeline Metrics</h3>
<div id="metrics"><table><thead><tr><th>Stage</th><th>Frames</th><th>Failed</th><th>Latency (us)</th><th>Queue</th></tr></thead><tbody id="metric-rows"></tbody></table></div>
<script>
async function refresh(){
 try{
  let s=await(await fetch('/api/status')).json();
  document.getElementById('cnt-ok').textContent=s.ok||0;
  document.getElementById('cnt-ng').textContent=s.ng||0;
  document.getElementById('cnt-warn').textContent=s.warn||0;
  document.getElementById('cnt-uncertain').textContent=s.uncertain||0;
  document.getElementById('uptime').textContent=(s.uptime_seconds||0).toFixed(0)+'s';
  let a=await(await fetch('/api/alerts')).json();
  let rows='';for(let r of a){rows+=`<tr><td>${r.frame_id}</td><td><span class="badge badge-${r.verdict.toLowerCase()}">${r.verdict}</span></td><td>${(r.severity||0).toFixed(2)}</td><td>${(r.defects||[]).join(', ')}</td></tr>`}
  document.getElementById('alert-rows').innerHTML=rows||'<tr><td colspan=4>No alerts</td></tr>';
  let m=await(await fetch('/api/metrics')).json();
  let mrows='';for(let s of m){mrows+=`<tr><td>${s.id}</td><td>${s.frames_processed}</td><td>${s.frames_failed}</td><td>${(s.avg_latency_us||0).toFixed(1)}</td><td>${s.queue_depth}</td></tr>`}
  document.getElementById('metric-rows').innerHTML=mrows||'<tr><td colspan=5>No metrics</td></tr>';
 }catch(e){console.error(e)}
}
refresh();setInterval(refresh,2000);
</script>
</body></html>)";
            res.set_content(html, "text/html; charset=utf-8");
        });

        svr_->listen("0.0.0.0", port_);
    });
}

void HttpServer::Stop() {
    if (!running_.exchange(false)) return;
    if (svr_) {
        svr_->stop();  // Unblock svr_->listen() so the thread can exit
    }
    thread_.request_stop();
    // std::jthread destructor auto-joins; no explicit join/detach needed
}

void HttpServer::UpdateResult(const InspectionSummary& summary) {
    std::lock_guard<std::mutex> lk(mutex_);
    history_.push_back(summary);
    if (history_.size() > kMaxHistory) {
        history_.erase(history_.begin(), history_.begin() + (history_.size() - kMaxHistory));
    }
    if (summary.verdict == "OK") ++ok_count_;
    else if (summary.verdict == "NG") ++ng_count_;
    else if (summary.verdict == "WARN") ++warn_count_;
    else ++uncertain_count_;
}

void HttpServer::UpdateMetrics(const std::vector<sai::pipeline::StageMetrics>& metrics) {
    std::lock_guard<std::mutex> lk(mutex_);
    metrics_.clear();
    metrics_.reserve(metrics.size());
    for (auto& m : metrics) {
        MetricsSnapshot snap;
        snap.stage_id = m.stage_id;
        snap.frames_processed = m.frames_processed.load();
        snap.frames_failed = m.frames_failed.load();
        snap.avg_latency_us = m.avg_latency_us.load();
        snap.queue_depth = m.queue_depth();
        metrics_.push_back(std::move(snap));
    }
}

void HttpServer::SetStartTime(std::chrono::steady_clock::time_point t) {
    start_time_ = t;
}

}  // namespace sai::web
