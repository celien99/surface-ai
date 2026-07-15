// opcua_client.h — OPC UA client for MES/PLC integration (open62541)
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sai/core/error.h>
#include <sai/plugin/plugin.h>
#include <sai/rule/value.h>

namespace sai::io {
struct InspectionResult;
}  // namespace sai::io

namespace sai::device {

// OPC UA variable subscription callback.
// Invoked when a subscribed variable changes value.
using OpcUaSubscriptionCallback = std::function<void(std::string_view node_id, sai::rule::Value value)>;

// IOpcUaClient: reads production context from MES/PLC via OPC UA.
//
// Implements IPlugin — lifecycle managed by Context (OnInitialize/OnStart/OnStop).
// Not an IDevice — OPC UA is a data bus, not a physical device.
//
// Typical OPC UA node mapping for seat leather AOI:
//   ns=2;s=BatchID         → current batch identifier (string)
//   ns=2;s=MaterialSKU     → material SKU (string)
//   ns=2;s=RejectRatePct   → historical batch reject rate (double)
//   ns=2;s=LineStatus      → production line status (string: "running"/"stopped")
//   ns=2;s=TotalUnits      → total units in batch (int64)
//   ns=2;s=InspectionResult → written back after each frame (string: JSON)
//
// Usage:
//   auto client = std::make_shared<OpcUaClient>("opc.tcp://192.168.1.100:4840");
//   client->OnInitialize(ctx);
//   client->OnStart(ctx);
//   auto batch_id = client->ReadValue("ns=2;s=BatchID");
//   client->WriteResult("ns=2;s=InspectionResult", inspection_result);
//   client->OnStop(ctx);

class IOpcUaClient : public IPlugin {
public:
    SAI_DECLARE_TYPE_ID(sai.device.opcua-client)

    // Read a single OPC UA variable value by NodeId string.
    [[nodiscard]] virtual auto ReadValue(std::string_view node_id) noexcept
        -> Result<sai::rule::Value> = 0;

    // Read multiple variables in one call (reduces round-trips).
    [[nodiscard]] virtual auto ReadValues(
        const std::vector<std::string>& node_ids) noexcept
        -> Result<std::vector<std::pair<std::string, sai::rule::Value>>> = 0;

    // Write detection result back to MES (typically as JSON string).
    [[nodiscard]] virtual auto WriteValue(
        std::string_view node_id, const sai::rule::Value& value) noexcept
        -> Result<void> = 0;

    // Subscribe to variable changes (push notifications from server).
    // The callback is invoked on the OPC UA client's internal thread.
    [[nodiscard]] virtual auto Subscribe(
        std::string_view node_id, OpcUaSubscriptionCallback callback) noexcept
        -> Result<void> = 0;

    // Check if the client is connected to the OPC UA server.
    [[nodiscard]] virtual auto IsConnected() const noexcept -> bool = 0;

    // IPlugin lifecycle
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override = 0;
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override = 0;
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

protected:
    PluginManifest manifest_{};
};

// OpcUaClient: concrete implementation using open62541.
// Linux-gated — open62541 is only available on Linux via vcpkg.
// On non-Linux platforms, a stub returning errors is provided.
class OpcUaClient final : public IOpcUaClient {
public:
    explicit OpcUaClient(std::string server_url);
    ~OpcUaClient() override;

    [[nodiscard]] auto ReadValue(std::string_view node_id) noexcept
        -> Result<sai::rule::Value> override;
    [[nodiscard]] auto ReadValues(
        const std::vector<std::string>& node_ids) noexcept
        -> Result<std::vector<std::pair<std::string, sai::rule::Value>>> override;
    [[nodiscard]] auto WriteValue(
        std::string_view node_id, const sai::rule::Value& value) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Subscribe(
        std::string_view node_id, OpcUaSubscriptionCallback callback) noexcept
        -> Result<void> override;
    [[nodiscard]] auto IsConnected() const noexcept -> bool override;

    [[nodiscard]] auto OnStart(Context&) -> Result<void> override;
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override;

    OpcUaClient(OpcUaClient&&) = delete;
    auto operator=(OpcUaClient&&) -> OpcUaClient& = delete;
    OpcUaClient(const OpcUaClient&) = delete;
    auto operator=(const OpcUaClient&) -> OpcUaClient& = delete;

private:
    std::string server_url_;
    void* client_ = nullptr;  // UA_Client* (opaque, avoids header leakage)
    bool connected_ = false;
};

}  // namespace sai::device
