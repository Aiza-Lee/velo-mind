#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velomind/macros.h"
#include "velomind/types.h"
#include "velomind_core_export.h"

namespace velomind {

enum class Op : std::uint16_t;

// 设备内存/显存快照
struct DeviceMemoryInfo {
    std::size_t total_bytes = 0;
    std::size_t used_bytes  = 0;
    std::size_t free_bytes  = 0;
};

// 统一硬件设备抽象
class VELOMIND_CORE_EXPORT Device {
public:
    constexpr Device() noexcept = default;
    constexpr Device(DeviceType type, int index = 0) noexcept
        : _type(type), _index(index) {}

    // 便捷工厂方法
    static constexpr auto cpu() noexcept -> Device { return Device(DeviceType::CPU, 0); }
    static constexpr auto ispc() noexcept -> Device { return Device(DeviceType::ISPC, 0); }
    static constexpr auto cuda(int index = 0) noexcept -> Device { return Device(DeviceType::CUDA, index); }
    static constexpr auto vulkan(int index = 0) noexcept -> Device { return Device(DeviceType::VULKAN, index); }

    // 字符串解析 (如 "cpu", "ispc", "cuda", "cuda:0", "vulkan")
    static auto from_string(std::string_view str) -> std::optional<Device>;

    // 机器设备自动探测与枚举
    static auto available_devices() -> std::vector<Device>;
    static auto default_device() -> Device;

    // 属性访问
    [[nodiscard]] constexpr auto type() const noexcept -> DeviceType { return _type; }
    [[nodiscard]] constexpr auto index() const noexcept -> int { return _index; }

    // 可用性与信息查询
    [[nodiscard]] auto is_available() const noexcept -> bool;
    [[nodiscard]] auto name() const -> std::string;
    [[nodiscard]] auto description() const -> std::string;

    // 能力与算子支持性查询
    [[nodiscard]] auto supports(DataType dtype) const noexcept -> bool;
    [[nodiscard]] auto supports(Op op, DataType dtype) const noexcept -> bool;
    [[nodiscard]] auto supports(
        Op op,
        std::span<const DataType> input_dtypes,
        DataType output_dtype
    ) const noexcept -> bool;

    // 显存 / 内存信息探测
    [[nodiscard]] auto memory_info() const -> std::optional<DeviceMemoryInfo>;

    // 隐式转换为底层 DeviceType 枚举
    constexpr operator DeviceType() const noexcept { return _type; }

    // 比较与排序支持
    auto operator==(const Device& other) const noexcept -> bool = default;
    auto operator<=>(const Device& other) const noexcept = default;

private:
    DeviceType _type{DeviceType::CPU};
    int        _index{0};
};

VELOMIND_CORE_EXPORT auto to_string(const Device& device) -> std::string;
VELOMIND_CORE_EXPORT auto operator<<(std::ostream& os, const Device& device) -> std::ostream&;

} // namespace velomind

template <>
struct std::hash<velomind::Device> {
    auto operator()(const velomind::Device& d) const noexcept -> std::size_t {
        return (static_cast<std::size_t>(d.type()) << 16) ^ static_cast<std::size_t>(d.index());
    }
};
