#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace velomind {

using index_t = std::int64_t;

using dim_t   = std::int64_t;

using shape_t   = std::vector<dim_t>;
using stride_t  = std::vector<dim_t>;
using strides_t = stride_t;

// 枚举值必须连续；Last 同时决定注册表容量。
enum class DataType : std::uint8_t {
    Float32,
    Int32,
    Int8,
    Bool,
    Float16,
    BFloat16,

    Last = BFloat16,
};

enum class DeviceType : std::uint8_t {
    CPU,
    CUDA,
    ISPC,
    VULKAN,

    Last = VULKAN,
};

constexpr std::size_t MAX_DEVICE_TYPES =
    static_cast<std::size_t>(DeviceType::Last) + 1;

constexpr std::size_t data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:  return sizeof(float);
        case DataType::Int32:    return sizeof(std::int32_t);
        case DataType::Int8:     return sizeof(std::int8_t);
        case DataType::Bool:     return sizeof(bool);
        case DataType::Float16:  return 2;
        case DataType::BFloat16: return 2;
    }
    return 0;
}

constexpr const char* data_type_name(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:  return "Float32";
        case DataType::Int32:    return "Int32";
        case DataType::Int8:     return "Int8";
        case DataType::Bool:     return "Bool";
        case DataType::Float16:  return "Float16";
        case DataType::BFloat16: return "BFloat16";
    }
    return "Unknown";
}

constexpr const char* device_type_name(DeviceType device) {
    switch (device) {
        case DeviceType::CPU:    return "CPU";
        case DeviceType::CUDA:   return "CUDA";
        case DeviceType::ISPC:   return "ISPC";
        case DeviceType::VULKAN: return "Vulkan";
    }
    return "Unknown";
}

// 格式化向量形状或步长为可读字符串，例如 "[2, 3, 4]"
inline auto to_string(const shape_t& shape) -> std::string {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

inline auto to_string(DataType dtype) -> std::string {
    return data_type_name(dtype);
}

inline auto to_string(DeviceType device) -> std::string {
    return device_type_name(device);
}

inline auto operator<<(std::ostream& os, DataType dtype) -> std::ostream& {
    return os << data_type_name(dtype);
}

inline auto operator<<(std::ostream& os, DeviceType device) -> std::ostream& {
    return os << device_type_name(device);
}

inline auto operator<<(std::ostream& os, const shape_t& shape) -> std::ostream& {
    return os << to_string(shape);
}

} // namespace velomind

#include "velomind/dtype.h"
