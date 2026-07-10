#pragma once

#include <filesystem>
#include <shared_mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <yaml-cpp/yaml.h>

#include <sai/core/error.h>
#include <sai/infra/config_schema.h>

namespace sai::infra {

class ConfigStore final {
public:
    explicit ConfigStore(ConfigSchema schema) noexcept;
    ~ConfigStore() noexcept;

    ConfigStore(const ConfigStore&) = delete;
    ConfigStore& operator=(const ConfigStore&) = delete;
    ConfigStore(ConfigStore&&) = delete;
    ConfigStore& operator=(ConfigStore&&) = delete;

    // 启动期一次性调用：读取 path 指向的 YAML 文件，路径不存在返回
    // Infra_ConfigFileNotFound、语法错误返回 Infra_ConfigParseError；解析成功后
    // 立即用构造时传入的 ConfigSchema 校验，校验失败返回 Infra_ConfigValidationFailed，
    // 且本次失败不修改内部已持有的配置树。
    [[nodiscard]] auto Load(std::filesystem::path path) -> Result<void>;

    // 按点分路径取出配置值并转换为 T；键不存在返回 Infra_ConfigKeyNotFound，
    // 存在但无法转换为 T 返回 Infra_ConfigKeyTypeMismatch。对配置树的访问加
    // shared_lock，与热重载线程的写锁互斥。
    template <typename T>
    [[nodiscard]] auto Get(std::string_view key) const -> Result<T> {
        std::shared_lock lock(mutex_);
        const YAML::Node node = detail::ResolveFieldPath(root_, key);
        if (!node.IsDefined()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigKeyNotFound,
                std::string("config key not found: ").append(key),
                std::source_location::current(),
            });
        }
        try {
            return node.as<T>();
        } catch (const YAML::Exception& ex) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigKeyTypeMismatch,
                std::string("config key type mismatch at '").append(key).append("': ") + ex.what(),
                std::source_location::current(),
            });
        }
    }

    // 启动 inotify 监听线程，监听 Load 时记录的文件路径；stop_token 用于随进程
    // 关停一并停止监听线程。本方法仅在 Linux 目标平台构建（Task 10 的
    // config_store_inotify.cpp 定义），本可移植子集只声明、不定义。
    [[nodiscard]] auto EnableHotReload(std::stop_token stop_token) -> Result<void>;

private:
    ConfigSchema schema_;
    std::filesystem::path path_;
    mutable std::shared_mutex mutex_;
    YAML::Node root_;              // 当前生效的、已通过校验的配置树
    std::jthread watch_thread_;    // 热重载监听线程（Task 10 启动）
};

}  // namespace sai::infra
