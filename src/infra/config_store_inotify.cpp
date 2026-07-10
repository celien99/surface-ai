// Linux-only definition of ConfigStore::EnableHotReload.
//
// This translation unit is gated to Linux by src/infra/CMakeLists.txt
// (`if(CMAKE_SYSTEM_NAME STREQUAL "Linux")`) and is NEVER compiled on the macOS
// arm64 dev host. It contains the real inotify implementation for the target
// platform (Ubuntu x64); there is deliberately no macOS FSEvents/kqueue fallback
// and no portable stub — on non-Linux builds EnableHotReload stays
// declared-but-undefined, which is fine because nothing in the portable subset
// ODR-uses it. See design doc 1.6-cross-cutting.md §3/§5/§8.
#include <sai/infra/config_store.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <source_location>
#include <string>

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/inotify.h>
#include <unistd.h>

#include <sai/infra/logger.h>

namespace sai::infra {

auto ConfigStore::EnableHotReload(std::stop_token stop_token) -> Result<void> {
    const int inotify_fd = ::inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
    if (inotify_fd < 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            std::string("inotify_init1 failed: ") + std::strerror(errno),
            std::source_location::current(),
        });
    }

    // Watch the PARENT DIRECTORY, not the file inode. Editors and atomic config
    // deploys save via write-to-temp + rename-over-target, which replaces the
    // inode; a watch bound to the old inode dies with IN_IGNORED and would never
    // fire again. A directory watch survives the replacement, so we watch the dir
    // for IN_CLOSE_WRITE (in-place writers) and IN_MOVED_TO (rename-over deploys)
    // and filter events by the target filename below.
    const std::filesystem::path dir = path_.parent_path();
    const std::string filename = path_.filename().string();
    const int watch_fd = ::inotify_add_watch(
        inotify_fd, dir.c_str(), IN_CLOSE_WRITE | IN_MOVED_TO);
    if (watch_fd < 0) {
        ::close(inotify_fd);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigFileNotFound,
            "inotify_add_watch on " + dir.string() + " failed: " + std::strerror(errno),
            std::source_location::current(),
        });
    }

    // A blocking read()/poll() cannot observe stop_token on its own, so we add an
    // eventfd to the poll set and have stop-callbacks write to it. This wakes the
    // watch thread immediately on shutdown (no poll timeout, no periodic
    // stop_requested() spin) — event-driven for shutdown just as inotify is
    // event-driven for reloads.
    const int stop_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (stop_fd < 0) {
        ::close(inotify_fd);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            std::string("eventfd failed: ") + std::strerror(errno),
            std::source_location::current(),
        });
    }

    watch_thread_ = std::jthread(
        [this, inotify_fd, stop_fd, filename, stop_token](std::stop_token thread_stop) {
            auto& log = Logger::Get("ConfigStore");

            // Re-read + re-parse + re-validate the file. root_ is replaced ONLY on
            // full success; on ANY failure (parse or validation) we log an Error
            // and leave root_ untouched — the store keeps serving the last
            // known-good config and NEVER falls back to defaults (design §3/§14).
            const auto reload = [this, &log]() {
                YAML::Node parsed;
                try {
                    parsed = YAML::LoadFile(path_.string());
                } catch (const YAML::BadFile& ex) {
                    log.Log(LogLevel::Error,
                            "hot-reload: config file unreadable, keeping previous config: {}",
                            ex.what());
                    return;
                } catch (const YAML::ParserException& ex) {
                    log.Log(LogLevel::Error,
                            "hot-reload: YAML parse failed, keeping previous config: {}",
                            ex.what());
                    return;
                } catch (const YAML::Exception& ex) {
                    // Any other yaml-cpp throw type must not escape the thread
                    // function (that would std::terminate the process). Keep stale.
                    log.Log(LogLevel::Error,
                            "hot-reload: unexpected YAML error, keeping previous config: {}",
                            ex.what());
                    return;
                }
                if (auto validated = schema_.Validate(parsed); !validated) {
                    log.Log(LogLevel::Error,
                            "hot-reload: validation failed, keeping previous config: {}",
                            validated.error().message);
                    return;
                }
                {
                    std::unique_lock lock(mutex_);
                    root_ = parsed;
                }
                log.Log(LogLevel::Info, "hot-reload: config reloaded from {}",
                        path_.string());
            };

            const auto signal_stop = [stop_fd]() noexcept {
                const std::uint64_t one = 1;
                if (::write(stop_fd, &one, sizeof(one)) < 0) {
                    // Nothing actionable: the eventfd counter is already non-zero
                    // (EAGAIN) or the fd is gone during teardown; poll wakes anyway.
                }
            };
            // Scope the stop-callbacks and the poll loop in an inner block so both
            // callbacks destruct (deregister, and block on any in-flight callback)
            // BEFORE the fds are closed below. Otherwise, after one token fires and
            // the loop exits, the OTHER token could fire for the first time and run
            // signal_stop's ::write() on an already-closed (possibly reused) stop_fd.
            {
                // Either the process-level stop_token (main() shutdown) or the
                // jthread's own stop_token (fired by watch_thread_'s destructor on
                // ConfigStore teardown) must wake and end the loop; listen to both.
                std::stop_callback ext_cb(stop_token, signal_stop);
                std::stop_callback thr_cb(thread_stop, signal_stop);

                std::array<pollfd, 2> fds{
                    pollfd{inotify_fd, POLLIN, 0},
                    pollfd{stop_fd, POLLIN, 0},
                };
                alignas(alignof(inotify_event)) std::array<std::byte, 4096> buf;

                while (!stop_token.stop_requested() && !thread_stop.stop_requested()) {
                    const int ready = ::poll(fds.data(), fds.size(), -1);
                    if (ready < 0) {
                        if (errno == EINTR) {
                            continue;
                        }
                        log.Log(LogLevel::Error, "hot-reload: poll failed, stopping watch: {}",
                                std::strerror(errno));
                        break;
                    }
                    if (fds[1].revents & POLLIN) {
                        break;  // stop signalled
                    }
                    if ((fds[0].revents | fds[1].revents) & (POLLERR | POLLHUP)) {
                        log.Log(LogLevel::Error,
                                "hot-reload: inotify fd error (POLLERR/POLLHUP), stopping watch");
                        break;
                    }
                    if (!(fds[0].revents & POLLIN)) {
                        continue;
                    }

                    const ssize_t len = ::read(inotify_fd, buf.data(), buf.size());
                    if (len <= 0) {
                        continue;
                    }

                    bool names_target = false;
                    for (std::byte* p = buf.data(); p < buf.data() + len;) {
                        const auto* ev = reinterpret_cast<const inotify_event*>(p);
                        if (ev->len > 0 && filename == ev->name) {
                            names_target = true;
                        }
                        p += sizeof(inotify_event) + ev->len;
                    }
                    if (names_target) {
                        reload();
                    }
                }
            }

            ::close(stop_fd);
            ::close(inotify_fd);
        });

    return {};
}

}  // namespace sai::infra
