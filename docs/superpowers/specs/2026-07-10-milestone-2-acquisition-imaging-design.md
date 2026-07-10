# Surface AI Framework —— 里程碑 2 采集与影像 设计文档

> Status: Draft
> Date: 2026-07-10
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 2
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）

---

## 1. 背景与范围

里程碑 2 覆盖"采集与影像"：从物理相机取帧，走完预处理链，产出 `SurfaceImage`，并通过可插拔的 IO 层导出检测结果。

### 1.1 里程碑 1 遗留项（本轮修复）

| 编号 | 遗留偏差 | 修复策略 |
|------|---------|---------|
| D1 | `GpuStreamQueue::EnqueueAsyncCopy` 从未填充/排空中转缓冲 | 在 2.2 预处理链的 GPU 路径中，由调用方显式 `std::memcpy` 填充 pinned buffer 后提交 HtoD，DtoH 完成后从 pinned buffer 构建 `SurfaceImage`。不修改 `GpuStreamQueue` 接口 |
| D2 | `Logger::DroppedCount()` 进程级而非 per-category | 将 `dropped_count_` 从单值改为 `std::unordered_map<std::string, std::atomic<uint64_t>>`，按 category 键控，保持现有 `DroppedCount()` 签名不变 |
| D3 | 日志轮转只有 size-only，缺 daily 条件 | 实现一个复合 sink：同时检查文件大小（100MB）和日期变更，任一条件触发即轮转。封装为 `DailyAndSizeRotatingFileSink` |

### 1.2 批次划分与执行顺序

```
里程碑 2：采集与影像
├── 2.1 Device（Camera + Light Controller）
├── 2.2 Imaging（图像类型体系 + 预处理链）
└── 2.3 IO（导入导出 + Exporter）
```

执行顺序：**2.1 设计 → 2.1 代码 → 2.2 设计 → 2.2 代码 → 2.3 设计 → 2.3 代码 → 最终整体回顾**

### 1.3 项目锚点

Surface AI 的首要落地场景是**工业生产流水线中的汽车座椅 AOI 缺陷检测**。关键约束：

- **多 SKU**：不同汽车厂商（奔驰、宝马、特斯拉等）的座椅型号不同，框架必须支持 SKU 级别的配置切换
- **多厂商**：座椅供应商不同，产线环境、照明条件、缺陷标准各异
- **光照不理想**：工业产线不是光学实验室，不可能提供均匀理想的光照。所有预处理和检测算法必须对此具有鲁棒性
- **程序不控制 PLC/MES**：程序仅与外部系统通信（读取状态、上报结果），不参与控制

### 1.4 明确排除项

- PLC/OPC UA 通信（外部系统通信，在本里程碑范围外）
- Robot/Encoder/Sensor/IO Board/DAQ 等更多设备类型（仅做 Camera + Light Controller）
- Polarization/Geometry Alignment/Image Pyramid/Streaming Image 等高级影像功能
- MES HTTP 上报（作为未来 Exporter 插件，本批次仅做 JSON + 文件导出）

---

## 2. 对里程碑 1 的依赖

| M2 批次 | 依赖 M1 接口 | 用途 |
|---------|-------------|------|
| 2.1 | `IPlugin`, `PluginManager` (1.3) | Camera/LightController 作为插件加载 |
| 2.1 | `Context` (1.2) | DI 注入设备配置 |
| 2.1 | `ConfigStore` (1.6) | 设备参数 YAML 配置 |
| 2.1 | `Logger` (1.6) | 设备连接/断连日志 |
| 2.2 | `IMemoryPool`, `GpuPool`, `PinnedPool`, `PooledPtr` (1.5) | 图像像素缓冲池化分配 |
| 2.2 | `GpuStreamQueue`, `CopyDirection` (1.4) | CPU↔GPU 异步数据搬运 |
| 2.2 | `WorkerPool` (1.4) | 预处理并行化 |
| 2.2 | `Result<T>`, `Resource` (1.1) | 错误处理 + 图像 RAII 基类 |
| 2.3 | `IPlugin`, `PluginManager` (1.3) | Exporter/Importer 作为插件加载 |
| 2.3 | `Result<T>` (1.1), `ConfigStore` (1.6), `Logger` (1.6) | 错误处理、导出配置、日志 |

---

## 3. 批次 2.1 Device

### 3.1 Purpose

定义框架直接控制的物理设备的统一抽象。本批次落地 Camera（GenICam/GigE Vision）和 Light Controller 两个设备类型。设计核心原则：工业产线光照不理想，Light Controller 接口适配现实硬件能力，不假设实验室级精确调光。

### 3.2 Responsibilities

- 提供 `IDevice` 统一设备基类，继承 `IPlugin`（1.3）
- `ICamera`：触发模式配置、采集启停、帧回调、曝光/增益/ROI 控制
- `ILightController`：多通道强度控制、频闪模式（跟随硬件触发）
- 设备发现：相机走 GenTL 枚举，光源控制器走配置文件显式声明

### 3.3 Design

**为什么 IDevice 继承 IPlugin 而非 IModule**：设备是动态库（`.so`）粒度的可插拔单元，走 `PluginManager` 的加载/卸载/生命周期通道。`IModule` 是编译期模块（在 1.2 中定义），设备不应绑定到编译期。

**为什么不做软件触发同步**：相机触发和光源频闪的同步若走软件路径（应用发命令 → 相机曝光 → 等完成 → 发命令 → 光源亮），延迟抖动在毫秒级，对高速产线不可接受。采用硬件触发（编码器/光电传感器 → 相机 Line 0 + 光源频闪输入端并联），框架只负责配置触发参数，不参与实时触发路径。

**为什么光照不理想**：工业产线不是光学实验室。光源可能老化、被遮挡、安装角度偏差、环境光干扰。`ILightController` 接口不假设"通道 0=环形光, 通道 1=背光"等固定语义——通道映射由 YAML 配置，实际光照效果由 2.2 的 FlatField/HDR 补偿。

**为什么 RingBuffer 满时丢弃最旧帧而非背压**：产线上帧来得比处理快是常态（相机 30fps，AI 推理可能 10fps）。背压（停采等待）会导致产线节拍被打乱，丢旧帧比延迟更可接受。

### 3.4 Interfaces

```cpp
// device.h — 设备统一接口
#pragma once
#include <chrono>
#include <functional>
#include <string_view>
#include <sai/plugin/plugin.h>

namespace sai::device {

class IDevice : public IPlugin {
public:
    enum class State { Disconnected, Connected, Acquiring, Error };

    [[nodiscard]] virtual auto Connect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Disconnect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto IsConnected() const noexcept -> bool = 0;
    [[nodiscard]] virtual auto SerialNumber() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto CurrentState() const noexcept -> State = 0;
};

}  // namespace sai::device
```

```cpp
// camera.h — 相机能力接口
#pragma once
#include <chrono>
#include <functional>
#include <sai/device/device.h>

namespace sai::device {

class RawImage;  // forward — 定义在 2.2

class ICamera : public IDevice {
public:
    enum class TriggerMode { Software, Hardware, FreeRun };

    [[nodiscard]] virtual auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto StartAcquisition() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto StopAcquisition() noexcept -> Result<void> = 0;

    // 回调在采集线程内调用，实现方必须立即返回，仅做 Push，不得阻塞。
    using FrameCallback = std::function<void(RawImage)>;
    [[nodiscard]] virtual auto RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> = 0;

    [[nodiscard]] virtual auto SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetGain(float db) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetROI(Rect region) noexcept -> Result<void> = 0;
};

}  // namespace sai::device
```

```cpp
// light_controller.h — 光源控制器接口
#pragma once
#include <sai/device/device.h>

namespace sai::device {

// 工业光源控制器通常是多通道物理盒子，每条产线上的光照条件不理想 —
// 此接口适配现实中的硬件能力，不支持假设具有实验室级别的均匀精确调光。
class ILightController : public IDevice {
public:
    enum class StrobeMode { Continuous, OnTrigger, Off };

    [[nodiscard]] virtual auto ChannelCount() const noexcept -> int = 0;
    [[nodiscard]] virtual auto SetIntensity(int channel, float intensity) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto GetIntensity(int channel) const noexcept -> Result<float> = 0;
    [[nodiscard]] virtual auto Enable(int channel) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Disable(int channel) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetStrobeMode(int channel, StrobeMode mode) noexcept -> Result<void> = 0;
};

}  // namespace sai::device
```

```cpp
// ring_buffer.h — 相机回调线程 → WorkerPool 线程之间的帧缓冲
#pragma once
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace sai::device {

// 固定容量的环形缓冲，满时覆盖最旧元素（见 3.3 Design：丢旧帧比背压更可接受）。
// 单生产者（FrameCallback，采集线程）、单/多消费者（WorkerPool 线程）均安全；
// T 要求可移动（RawImage 等 Image 子类满足）。容量在构造时固定，运行期不重新分配。
template <typename T>
class RingBuffer final {
public:
    explicit RingBuffer(std::size_t capacity) noexcept;

    // 回调线程内调用：capacity 已满时覆盖最旧的未消费元素并递增 dropped_count_，
    // 不阻塞、不返回错误——本类不对外暴露背压路径。
    auto Push(T item) noexcept -> void;

    // Worker 线程内调用：缓冲为空返回 std::nullopt，不阻塞等待。
    [[nodiscard]] auto TryPop() noexcept -> std::optional<T>;

    [[nodiscard]] auto Capacity() const noexcept -> std::size_t { return capacity_; }
    [[nodiscard]] auto Size() const noexcept -> std::size_t;
    [[nodiscard]] auto DroppedCount() const noexcept -> std::size_t { return dropped_count_; }

    RingBuffer(const RingBuffer&) = delete;
    auto operator=(const RingBuffer&) -> RingBuffer& = delete;

private:
    std::vector<std::optional<T>> slots_;
    std::size_t capacity_;
    std::size_t head_ = 0;  // 下一个 Push 写入位
    std::size_t tail_ = 0;  // 下一个 TryPop 读取位
    std::size_t count_ = 0;
    std::size_t dropped_count_ = 0;
    mutable std::mutex mutex_;  // 容量小（帧级，通常 <= 8），互斥锁足够，不引入无锁结构
};

}  // namespace sai::device
```

### 3.5 Workflow

```
1. PluginManager::Load("genican_camera.so")
     → 相机插件 OnInitialize（解析配置，枚举 GenTL 设备列表）
     → OnStart（无实际操作，设备尚未连接）

2. camera.Connect()
     → 打开 GenTL 设备句柄
     → 设置像素格式、包大小

3. camera.SetTriggerMode(Hardware)
     → 配置 Line 0 为触发输入
     → SetExposureTime / SetGain / SetROI

4. camera.RegisterFrameCallback(RingBuffer::Push)
     → 注册回调，回调中仅做无锁 push

5. camera.StartAcquisition()
     → 相机进入等待触发状态
     → 外部编码器触发 → 曝光 → 回调 → RingBuffer

6. camera.StopAcquisition()
     → 停止流，清空驱动缓冲区

7. camera.Disconnect()
     → 释放 GenTL 句柄
```

连接/采集状态机：`Disconnected → Connected → Acquiring → Connected → Disconnected`

### 3.6 Data Structure

```cpp
// Rect — 定义于 device.h（IDevice/ICamera 同一头文件内，故 3.4 的 camera.h 通过其
// #include <sai/device/device.h> 已可见 Rect，无需单独 include）；设备层和影像层共用，
// 4.6 的 sai::image::Rect 是本类型的 using 别名。
struct Rect {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t width = 0;
    std::size_t height = 0;

    [[nodiscard]] auto Area() const noexcept -> std::size_t { return width * height; }
    [[nodiscard]] auto IsEmpty() const noexcept -> bool { return width == 0 || height == 0; }
};

// 设备配置 YAML 示例
// device:
//   camera:
//     trigger_mode: hardware
//     exposure_us: 500
//     gain_db: 0.0
//     roi: { x: 0, y: 0, width: 2448, height: 2048 }
//     pixel_format: Mono8
//   light_controller:
//     serial_port: /dev/ttyUSB0
//     baud_rate: 115200
//     channels:
//       - { id: 0, intensity: 0.8, strobe: on_trigger }
//       - { id: 1, intensity: 0.5, strobe: continuous }
```

### 3.7 Class Diagram

```
IReflectable (1.1)  ←  IModule (1.2)  ←  IPlugin (1.3)  ←  IDevice (2.1)
                                                                │
                                               ┌────────────────┼────────────────┐
                                               │                                  │
                                          ICamera (2.1)              ILightController (2.1, final)
```

### 3.8 Sequence Diagram

```
User/App → PluginManager::Load → OnInitialize(ConfigStore) → 枚举 GenTL 设备
         → camera.Connect() → 打开设备句柄
         → SetTriggerMode(Hardware) → SetExposureTime → RegisterFrameCallback
         → StartAcquisition → [硬件触发 → 曝光 → FrameCallback → RingBuffer]
         → StopAcquisition → Disconnect
```

### 3.9 Thread Model

| 线程 | 归属 | 职责 |
|------|------|------|
| GenTL 采集线程 | 相机 SDK 内部 | 等硬件触发、取帧、调 `FrameCallback`、立即返回 |
| 主线程 | 应用入口 | 所有控制面操作（Connect/Start/Stop）串行化 |
| LightController IO 线程 | 插件内部 | 串口 read/write，插件自行管理 |

规则：`Connect()`/`Disconnect()` 不与采集并行调用（State 机保证）；采集参数在采集停止后修改。

### 3.10 Performance

| 指标 | 目标 |
|------|------|
| Connect() 延迟 | < 500ms |
| 回调内耗时 | < 1μs（仅 `RingBuffer::Push`） |
| 帧丢失率 | 产线触发频率 ≤ 标称帧率时 0%；超出时丢弃最旧帧 |
| 曝光/增益设置 | < 10ms（寄存器写入） |

### 3.11 Memory

- `RawImage` 像素缓冲从 `IMemoryPool`（PinnedPool）分配，回调传移动语义
- 设备对象由 `PluginManager` 持有的 `shared_ptr<IPlugin>` 管理生命周期
- `DestroyPluginFn` 负责析构，禁止跨 `.so` 边界 `delete`

### 3.12 Future Extension

- 更多设备类型（Robot、Encoder、Sensor）实现 `IDevice` 或子接口，不修改已有代码
- 多相机同步：通过 GenTL IEEE 1588 PTP 硬件级同步
- FreeRun 模式作为 `TriggerMode::FreeRun` 的扩展

### 3.13 Best Practice

- 设备配置一律走 `ConfigStore`，不硬编码曝光/增益/端口
- 回调只做转发，不在回调内做任何图像处理
- 设备断连后必须 `Disconnect` → `Connect` 完整重连，不允许部分恢复

### 3.14 Anti Pattern

- ❌ 在 `FrameCallback` 中做 `spdlog::info` 或图像处理（阻塞采集线程 → 丢帧）
- ❌ 假设所有相机品牌行为一致——GenICam 标准功能每个厂商实现质量不同，必须用实际硬件测试
- ❌ 假设产线有理想光源——光照不均匀是常态，不要在 LightController 层假设固定通道语义

---

## 4. 批次 2.2 Imaging

### 4.1 Purpose

定义图像类型体系（RawImage → SurfaceImage → GpuImage）和可组合的预处理链。修复 M1 遗留的 D1（GPU 数据路径缺失 populate/drain）。

### 4.2 Responsibilities

- `Image` 基类：继承 `Resource`（1.1），像素缓冲独占所有权，只移动不拷贝
- `RawImage`：相机原始帧，可能 Bayer 编码、未校正
- `SurfaceImage`：预处理完毕，RGB8/Mono8，可直接喂入 AI
- `GpuImage`：显存驻留，零拷贝衔接 TensorRT
- `PreprocessFn`：自由函数管道，编译期组合，运行时顺序执行
- 内置步骤：HDR、FlatField、Debayer、Calibration、WhiteBalance、Resize

### 4.3 Design

**为什么 Image 继承 Resource 而非 Object**：图像是 RAII 资源——像素缓冲独占所有权，只移动不拷贝。`Resource`（1.1）的移动语义天然匹配。`Object` 是禁止移动/拷贝的纯基类，不适合持有大块内存的实体。

**为什么不用虚函数做预处理步骤**：预处理链的步骤顺序是启动期由配置确定的，运行时不发生多态切换。`std::function`（一次 indirect call ~2ns）相比虚函数（两次 indirect call ~5ns）更轻量。实际瓶颈在像素运算本身（毫秒级），调用开销差异可忽略。

**为什么不引入 GrayImage/RGBImage 等子类型**：颜色空间通过 `PixelFormat` 枚举表达即可。若每种像素格式都做子类，组合爆炸（BayerRG8/BayerRG12/Mono8/Mono12/RGB8/BGR8 × Raw/Surface/Gpu = 18 种类型），接口不统一。

**为什么 FlatField 在 Debayer 之前执行**：Bayer 模式下每个像素只记录一种颜色分量（R/G/B），Debayer 用邻域插值补全。先做平场校正（乘性校正每个像素的原始值）再 Debayer，比先插值再校正更精确——因为校正的是传感器原始信号而非插值估计值。

**D1 修复方案**：`GpuStreamQueue::EnqueueAsyncCopy` 负责传输但不负责填充/排空中转缓冲——这是通用传输层的正确设计。调用方（预处理链的 GPU 步骤）显式处理：

```cpp
// HtoD 路径
auto transit = pinned_pool.Acquire(size);       // 获取中转缓冲
std::memcpy(transit.Get(), cpu_image.Data(), size); // ← 填充（调用方职责）
co_await gpu_queue.EnqueueAsyncCopy(transit, size, HostToDevice, token);

// DtoH 路径
co_await gpu_queue.EnqueueAsyncCopy(transit, size, DeviceToHost, token);
auto result = SurfaceImage::FromPinned(std::move(transit), meta); // ← 排空（调用方职责）
```

### 4.4 Interfaces

```cpp
// image.h — Image 类型基类
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <sai/core/error.h>
#include <sai/core/resource.h>

namespace sai::image {

enum class PixelFormat : std::uint16_t {
    Mono8, Mono10, Mono12,
    BayerRG8, BayerRG10, BayerRG12,
    RGB8, BGR8,
    Undefined = 0xFFFF,
};

struct ImageMeta {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t channels = 0;
    PixelFormat pixel_format = PixelFormat::Undefined;
    std::chrono::nanoseconds timestamp{0};
    std::uint32_t frame_index = 0;
};

// Image 继承 Resource（1.1）——像素缓冲独占所有权，移动不拷贝。
// 抽象基类：构造函数 protected，只能通过 RawImage/SurfaceImage/GpuImage 的具名工厂函数间接构造，
// 跨类型的通用消费路径（PreprocessFn/ROI::Apply/IImporter::ImportImage）统一以
// std::unique_ptr<Image> 传递，避免按值传递切片丢失子类私有的 owner_pool_。
class Image : public Resource {
public:
    [[nodiscard]] auto Meta() const noexcept -> const ImageMeta& { return meta_; }
    [[nodiscard]] auto Data() const noexcept -> const std::uint8_t* { return data_; }
    [[nodiscard]] auto Data() noexcept -> std::uint8_t* { return data_; }
    [[nodiscard]] auto SizeBytes() const noexcept -> std::size_t { return size_bytes_; }

    // Resource（1.1）纯虚契约：data_ != nullptr 即视为持有有效缓冲。
    [[nodiscard]] auto IsValid() const noexcept -> bool override { return data_ != nullptr; }
    auto Release() noexcept -> void override;

protected:
    Image(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept;
    std::uint8_t* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    ImageMeta meta_{};
};

}  // namespace sai::image
```

```cpp
// raw_image.h — 相机原始输出
#pragma once
#include <sai/image/image.h>
#include <sai/memory/memory_pool.h>

namespace sai::image {

class RawImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& pool, ImageMeta meta) noexcept
        -> Result<RawImage>;
    // 调用方自持有缓冲（例如测试夹具/内存映射文件），本实例不持有 PooledPtr，
    // 析构时不触发任何池归还——data 的生命周期完全由调用方管理。
    [[nodiscard]] static auto FromBuffer(std::uint8_t* data, std::size_t size_bytes,
                                         ImageMeta meta) noexcept -> RawImage;

    RawImage(RawImage&&) noexcept = default;
    auto operator=(RawImage&&) noexcept -> RawImage& = default;
    ~RawImage() override;
    RawImage(const RawImage&) = delete;
    auto operator=(const RawImage&) -> RawImage& = delete;

private:
    // 持有实际的 PooledPtr<uint8_t> 句柄（而非裸 IMemoryPool* + 裸数据指针）——
    // IMemoryPool::Release 要求传入具体句柄以定位 slab 元数据中的引用计数槽，
    // 裸指针无法归还给池；FromBuffer 路径下 buffer_ 保持默认空句柄。
    explicit RawImage(sai::memory::PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept;
    RawImage(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept;  // FromBuffer 专用
    sai::memory::PooledPtr<std::uint8_t> buffer_{};
};

}  // namespace sai::image
```

```cpp
// surface_image.h — 预处理完毕的图像
#pragma once
#include <sai/image/image.h>

namespace sai::image {

class SurfaceImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& pool, ImageMeta meta) noexcept
        -> Result<SurfaceImage>;
    [[nodiscard]] static auto FromPinned(PooledPtr<std::uint8_t> pinned, ImageMeta meta) noexcept
        -> SurfaceImage;

    SurfaceImage(SurfaceImage&&) noexcept = default;
    auto operator=(SurfaceImage&&) noexcept -> SurfaceImage& = default;
    ~SurfaceImage() override;
    SurfaceImage(const SurfaceImage&) = delete;
    auto operator=(const SurfaceImage&) -> SurfaceImage& = delete;

private:
    // 见 raw_image.h 同名注释：持有 PooledPtr<uint8_t> 而非裸指针，析构自动归还池。
    explicit SurfaceImage(sai::memory::PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept;
    sai::memory::PooledPtr<std::uint8_t> buffer_{};
};

}  // namespace sai::image
```

```cpp
// gpu_image.h — 显存驻留图像
#pragma once
#include <sai/image/image.h>

namespace sai::image {

class GpuImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& gpu_pool, ImageMeta meta) noexcept
        -> Result<GpuImage>;

    GpuImage(GpuImage&&) noexcept = default;
    auto operator=(GpuImage&&) noexcept -> GpuImage& = default;
    ~GpuImage() override;
    GpuImage(const GpuImage&) = delete;
    auto operator=(const GpuImage&) -> GpuImage& = delete;

private:
    // 见 raw_image.h 同名注释：持有 PooledPtr<uint8_t>（底层为 GpuPool 分配的设备内存），
    // 析构自动归还池，不裸持有 IMemoryPool 指针。
    explicit GpuImage(sai::memory::PooledPtr<std::uint8_t> device_buffer, ImageMeta meta) noexcept;
    sai::memory::PooledPtr<std::uint8_t> buffer_{};
};

}  // namespace sai::image
```

```cpp
// preprocess.h — 预处理步骤
#pragma once
#include <functional>
#include <memory>
#include <vector>
#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::image {

// 预处理步骤：unique_ptr<Image> → unique_ptr<Image>，失败返回错误。
// 按 unique_ptr 传递而非按值传递 Image——Image 是抽象基类且构造函数 protected，
// 按值传递会切片并丢失 RawImage/SurfaceImage/GpuImage 各自持有的 PooledPtr<uint8_t>
// （池归还所必需），unique_ptr<Image> 保留动态类型与所有权，无切片风险。
using PreprocessFn =
    std::function<auto(std::unique_ptr<Image>) -> Result<std::unique_ptr<Image>>>;

// 将 steps 顺序串联，任一步骤失败则短路返回
[[nodiscard]] auto Compose(std::vector<PreprocessFn> steps) -> PreprocessFn;

// 内置步骤
[[nodiscard]] auto MakeDebayer() -> PreprocessFn;
[[nodiscard]] auto MakeFlatField(Image correction_frame) -> PreprocessFn;

struct CalibrationParams {
    std::array<double, 9> camera_matrix;
    std::array<double, 5> dist_coeffs;
    double pixel_scale_mm = 1.0;
};
[[nodiscard]] auto MakeCalibration(CalibrationParams params) -> PreprocessFn;
[[nodiscard]] auto MakeWhiteBalance(float r_gain, float g_gain, float b_gain) -> PreprocessFn;
[[nodiscard]] auto MakeHDR(std::size_t num_exposures) -> PreprocessFn;
[[nodiscard]] auto MakeResize(std::size_t target_width, std::size_t target_height) -> PreprocessFn;

}  // namespace sai::image
```

### 4.5 Workflow

```
Camera 回调 → RawImage (BayerRG12, 2448×2048, PinnedPool)
  → RingBuffer → WorkerPool 线程取出
  → HDR（可选，多帧融合）
  → FlatField（平场校正，补偿光照不均匀 + 镜头渐晕）
  → Debayer（Bayer → RGB8，Bayer 相机才需要）
  → Calibration（去畸变 + 物理尺度 mm/pixel）
  → WhiteBalance（灰卡白平衡）
  → Resize（缩放到 AI 输入尺寸 224×224）
  → [可选 GPU 路径] GpuImage（HtoD）→ GPU 处理 → SurfaceImage（DtoH）
  → SurfaceImage → 传给 2.3 IO 或 M3 AI
```

### 4.6 Data Structure

```cpp
// roi.h
namespace sai::image {

using Rect = sai::device::Rect;  // 共用定义

struct ROI {
    std::vector<Rect> regions;
    [[nodiscard]] auto IsEmpty() const noexcept -> bool { return regions.empty(); }
    [[nodiscard]] auto BoundingBox() const noexcept -> Rect;
    [[nodiscard]] static auto Apply(const Image& src, const ROI& roi)
        -> Result<std::unique_ptr<Image>>;
};

}  // namespace sai::image
```

### 4.7 Class Diagram

```
Resource (1.1, move-only)
  └── Image ──────────────────────────── width/height/channels/pixel_format/timestamp/frame_index
        ├── RawImage (final) ─────────── 相机原始帧，Bayer 可能
        ├── SurfaceImage (final) ─────── 预处理完毕，RGB8/Mono8
        └── GpuImage (final) ─────────── 显存驻留

独立类型：
  ImageMeta ────── POD 元数据
  PixelFormat ──── enum（与 GenICam PFNC 对齐）
  ROI ──────────── vector<Rect>
  PreprocessFn ─── std::function<auto(Image) -> Result<Image>>
  CalibrationParams ─ 相机内参 + 畸变系数 + pixel_scale_mm
```

### 4.8 Sequence Diagram

```
WorkerPool 线程 → PreprocessChain:
  Compose(cfg) 生成的链: HDR? → FlatField → Debayer → Calibration → WhiteBalance → Resize
  [GPU 分支]:
    PinnedPool::Acquire → memcpy(pinned, cpu_data) → EnqueueAsyncCopy(HtoD) → co_await
    → GpuPool::Acquire → [GPU 操作放 M3] → EnqueueAsyncCopy(DtoH) → co_await
    → SurfaceImage::FromPinned(pinned)
  → SurfaceImage → 传给下游
```

### 4.9 Thread Model

| 场景 | 线程 | 说明 |
|------|------|------|
| CPU 预处理链 | WorkerPool 线程 | 整条链在一个线程上同步执行 |
| GPU 路径 | WorkerPool 线程 + GPU Callback 线程 | 协程挂起/恢复，不阻塞 Worker |
| 多帧并行 | 多个 WorkerPool 线程 | 每帧一个 Worker，多帧并行 |

### 4.10 Performance

| 指标 | 目标 |
|------|------|
| CPU 预处理链总延迟（RawImage→SurfaceImage，不含 HDR） | < 5ms @ 2448×2048 Mono8 |
| Debayer（BayerRG12→RGB8, 2448×2048） | < 2ms |
| FlatField（乘性校正, 2448×2048） | < 1ms |
| Calibration（双线性插值去畸变） | < 3ms |
| Resize（2448→224, 双线性） | < 0.5ms |
| HtoD 异步拷贝（224×224×3, PCIe 3.0） | < 200μs |
| 内存峰值（单帧 2448×2048×3 + 双缓冲） | < 30MB |

### 4.11 Memory

| 对象 | 分配来源 | 生命周期 |
|------|---------|---------|
| RawImage 像素缓冲 | PinnedPool（相机插件分配） | 回调移交 → 预处理链消费完即释放 |
| 预处理中间产物 | IMemoryPool（CPU 池） | 当前步骤用完后归还 |
| GpuImage | GpuPool | 上传完成后中间 GPU buffer 可释放 |
| SurfaceImage | PinnedPool 或 CPU 池 | 传递给下游，由消费者释放 |
| 零拷贝 | — | 全程传指针，唯一拷贝点：Debayer、Resize、HtoD/DtoH |

### 4.12 Future Extension

- Polarization：`PixelFormat::Polarized*` + `MakePolarization()` 预处理步骤
- Image Pyramid：`MakePyramid()` 生成多尺度 `vector<Image>`
- Streaming Image：`GpuImage` 连续帧序列 + `RingBuffer<GpuImage>`
- Geometry Alignment：独立 `PreprocessFn`，配准模版图与实采图

### 4.13 Best Practice

- ✅ 预处理链配置通过 `ConfigStore` 的 YAML 加载，不硬编码顺序
- ✅ 每一步用 `Result<Image>` 传播错误——一步失败，整链短路
- ✅ 工业现场 FlatField 应在 Debayer **之前**执行（校正原始信号比校正插值更准确）
- ✅ `RawImage` 的 PinnedPool 分配应与相机 ROI 匹配——只分配实际用到的像素量

### 4.14 Anti Pattern

- ❌ 假设所有相机输出 RGB8——大部分工业相机是 Bayer 或 Mono
- ❌ 在预处理链中写磁盘或打日志——每帧都打 I/O 不可接受
- ❌ 忽略 `pixel_scale_mm`——没有物理尺度无法实现"缺陷 > 0.5mm 才报警"
- ❌ 在 HDR 不必要时仍然开 HDR——多帧融合增加 latency
- ❌ 假定实验室级均匀光照——`FlatField` 必须是默认开启的步骤

---

## 5. 批次 2.3 IO

### 5.1 Purpose

定义导入导出的可插拔接口。导出端产出检测报告（JSON + 标注图像），导入端加载标定文件和 SKU 配置。首要场景：汽车座椅 AOI 多 SKU 产线，不同厂商可能需要不同的导出格式。

### 5.2 Responsibilities

- `IExporter` 插件接口：导出 `InspectionResult` + 可选的标注图像
- `IImporter` 插件接口：导入标定帧/参考图像/SKU 配置 YAML
- 内置 `JsonExporter`（JSON 报告 + PNG）和 `BasicImporter`（PNG 图像 + YAML 元数据）
- 数据结构：`DefectRecord`、`InspectionResult`（由 M5 Reasoner 产出，2.3 定义）

### 5.3 Design

**为什么 Exporter/Importer 是插件**：不同汽车厂商（奔驰 vs 特斯拉 vs 宝马）要求不同的缺陷报告格式。丰田要 CSV、特斯拉要 JSON、奔驰要直连 MES。插件化让厂商格式成为可替换模块，框架核心不感知具体格式。

**为什么数据结构由 2.3 定义而非 M5**：`DefectRecord` 和 `InspectionResult` 是 IO 层的契约——谁产出、谁消费都基于此。M5 Reasoner 产出数据时依赖 2.3 的结构定义，2.3 不依赖 M5。

**为什么导出路径按 SKU + 序列号组织**：`output/{sku_id}/{serial_number}/` 的目录结构天然支持多 SKU 产线的追溯需求。

### 5.4 Interfaces

```cpp
// exporter.h — 导出插件接口
#pragma once
#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <sai/core/error.h>
#include <sai/device/device.h>  // Rect
#include <sai/image/surface_image.h>
#include <sai/plugin/plugin.h>

namespace sai::io {

struct DefectRecord {
    std::string label;            // "划痕", "破洞", "褶皱"
    std::string severity;         // "CRITICAL" | "MAJOR" | "MINOR"
    float confidence = 0.0F;
    Rect location;
    std::string evidence_path;   // 裁剪缺陷子图路径（可选）
};

struct InspectionResult {
    std::string sku_id;          // "Tesla_Model3_Driver"
    std::string serial_number;
    std::chrono::system_clock::time_point timestamp;
    std::vector<DefectRecord> defects;
    std::string verdict;         // "PASS" | "FAIL" | "RECHECK"
};

class IExporter : public IPlugin {
public:
    [[nodiscard]] virtual auto Export(const InspectionResult& result,
                                      std::filesystem::path output_dir,
                                      const SurfaceImage* annotated_image) noexcept
        -> Result<void> = 0;
    [[nodiscard]] virtual auto FormatName() const noexcept -> std::string_view = 0;
};

}  // namespace sai::io
```

```cpp
// importer.h — 导入插件接口
#pragma once
#include <filesystem>
#include <memory>
#include <string_view>
#include <yaml-cpp/yaml.h>
#include <sai/core/error.h>
#include <sai/image/image.h>
#include <sai/plugin/plugin.h>

namespace sai::io {

class IImporter : public IPlugin {
public:
    [[nodiscard]] virtual auto ImportImage(std::filesystem::path file_path) noexcept
        -> Result<std::unique_ptr<Image>> = 0;
    [[nodiscard]] virtual auto ImportMetadata(std::filesystem::path file_path) noexcept
        -> Result<YAML::Node> = 0;
    [[nodiscard]] virtual auto FormatName() const noexcept -> std::string_view = 0;
};

}  // namespace sai::io
```

```cpp
// json_exporter.h — 默认 JSON 导出实现
namespace sai::io {

class JsonExporter final : public IExporter {
public:
    [[nodiscard]] auto Export(const InspectionResult& result,
                              std::filesystem::path output_dir,
                              const SurfaceImage* annotated_image) noexcept
        -> Result<void> override;
    [[nodiscard]] auto FormatName() const noexcept -> std::string_view override
    { return "json_report"; }

    // Lifecycle（无状态导出器，直接通过）
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

private:
    PluginManifest manifest_{};
};

// basic_importer.h — 默认导入实现
class BasicImporter final : public IImporter {
public:
    [[nodiscard]] auto ImportImage(std::filesystem::path) noexcept
        -> Result<std::unique_ptr<Image>> override;
    [[nodiscard]] auto ImportMetadata(std::filesystem::path) noexcept -> Result<YAML::Node> override;
    [[nodiscard]] auto FormatName() const noexcept -> std::string_view override
    { return "basic_import"; }

    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

private:
    PluginManifest manifest_{};
};

}  // namespace sai::io
```

### 5.5 Workflow

```
M5 Reasoner → InspectionResult(defects, verdict, sku_id, serial_number)
  → IExporter::Export(result, output_dir, annotated_image)
    ├── 创建 output_dir/{sku_id}/{serial_number}/
    ├── 写 result.json（InspectionResult → JSON 序列化）
    ├── [如果有图] 写 annotated.png
    └── [可选 MES 上报] HTTP POST（未来插件）
  → Result<void>
```

### 5.6 Data Structure

见 §5.4 中的 `DefectRecord` 和 `InspectionResult`。

### 5.7 Class Diagram

```
IPlugin (1.3)
  ├── IExporter (2.3)
  │     ├── JsonExporter (final)
  │     └── ... 厂商自定义 Exporter
  └── IImporter (2.3)
        ├── BasicImporter (final)
        └── ... 厂商自定义 Importer

独立类型：
  DefectRecord ─────── label/severity/confidence/location/evidence_path
  InspectionResult ─── sku_id/serial/timestamp/defects/verdict
```

### 5.8 Sequence Diagram

```
Pipeline Worker 线程
  → IExporter::Export(result, output_dir, annotated)
    → ensure_directory(output_dir/sku_id/serial)
    → serialize_json(result) → write(result.json)
    → [if annotated] encode_png(annotated) → write(annotated.png)
    → [future MES] co_await http_post(...)
    → return Result<void>
```

### 5.9 Thread Model

| 操作 | 线程 | 说明 |
|------|------|------|
| Export() | Pipeline Worker 线程 | 导出是 Pipeline 最后一级，与检测流程串行 |
| 文件 IO | 调用线程内同步 | 输出量小（每帧 KB 级 JSON + 几百 KB PNG），不做异步 IO |
| MES HTTP（未来） | 调用线程异步（co_await） | 网络 IO 不阻塞 Pipeline |

### 5.10 Performance

| 指标 | 目标 |
|------|------|
| JSON 序列化（100 defects） | < 1ms |
| PNG 编码（224×224, RGB8） | < 5ms |
| 导出总延迟（JSON + PNG） | < 10ms |
| 导入标定帧（2448×2048 PNG） | < 50ms |
| 导入 SKU 配置 YAML | < 5ms |

### 5.11 Memory

| 对象 | 分配来源 | 生命周期 |
|------|---------|---------|
| InspectionResult | 栈/堆（由 M5 Reasoner 产出） | 传给 Export 后消费，调用返回后释放 |
| JSON 字符串 | 栈（std::string） | 写文件后即释放 |
| 导出图像 | 共享 SurfaceImage 像素缓冲 | 调用方持有，导出器只读不拷贝 |
| 导入图像 | 堆（Image 新建） | 返回给调用方，调用方负责释放 |

### 5.12 Future Extension

- CSV/Excel 导出：`CsvExporter` 插件
- MES HTTP 上报：`MesHttpExporter` 插件，直连工厂 MES
- 数据库直写：`SqliteExporter`，写入本地 SQLite（M4 Knowledge 也用）
- 多格式导出串联：同一份 `InspectionResult` 同时传给多个 Exporter

### 5.13 Best Practice

- ✅ 导出路径按 SKU + 序列号组织：`output/{sku_id}/{serial}/result.json`
- ✅ 每批次检测结果落地到独立目录，由外部清理策略管理
- ✅ 导入的标定数据在 `OnInitialize` 加载一次并缓存，运行期不反复读文件
- ✅ IExporter 插件可串联——同一份结果可同时交给 JsonExporter 和 MesHttpExporter

### 5.14 Anti Pattern

- ❌ 在导出器内做图像处理——导出器只负责序列化和写盘，不做标注框绘制
- ❌ 在热路径上打开/关闭文件——`OnStart` 中预打开句柄，运行期只 write
- ❌ 假设所有厂商需要同一种格式——必须可插拔
- ❌ 导出失败不报警——`Export` 返回错误时，调用方必须 log Error + 触发告警

---

## 6. 验证点

> GenICam 相机采一帧 → 预处理链 → SurfaceImage → JsonExporter 导出 result.json + annotated.png

验证步骤：

1. 加载 GenICam 相机插件 → Connect → 配置硬件触发 → StartAcquisition
2. 模拟一次触发（或软件触发）→ 回调收到 RawImage
3. RawImage 走完预处理链产出 SurfaceImage
4. 构造 InspectionResult 传给 JsonExporter
5. 检查 `output/{sku_id}/{serial}/` 下的 result.json 和 annotated.png

---

## 7. D2/D3 修复方案

### D2: per-category DroppedCount

```cpp
// logger.h 变更
class Logger final {
    // ...
    [[nodiscard]] auto DroppedCount() const noexcept -> uint64_t;  // 保持签名不变
    [[nodiscard]] auto DroppedCount(std::string_view category) const noexcept -> uint64_t;  // 新增重载

private:
    // 原来: std::atomic<uint64_t> dropped_count_{0};
    std::unordered_map<std::string, std::atomic<uint64_t>> category_dropped_counts_{};
};
```

### D3: Daily + Size 复合轮转

```cpp
// daily_and_size_sink.h — 新增
// 继承 spdlog::sinks::base_sink，在 _sink_it 中同时检查：
//   1. 当前文件大小 > max_size（100MB）
//   2. 当前文件创建日期 != 今天
// 任一条件触发即轮转。
```

---

## 8. 与后续里程碑的关系

- M3（AI 推理核心）：接收 2.2 的 `SurfaceImage`/`GpuImage` 作为推理输入
- M4（知识与检索）：使用 2.3 的 `IImporter` 加载参考图像和知识数据
- M5（推理决策）：产出的 `InspectionResult` 结构由 2.3 定义
- M6（编排调度）：2.1 的采集 → 2.2 的预处理 → M3 的推理 → 2.3 的导出，形成完整 Pipeline
