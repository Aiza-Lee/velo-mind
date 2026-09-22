#include "velomind/device.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

#include "velomind/dtype.h"
#include "internal/registry/storage.h"

#if defined(VELOMIND_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace velomind {

auto Device::from_string(std::string_view str) -> std::optional<Device> {
    if (str.empty()) return std::nullopt;

    auto colon = str.find(':');
    std::string_view base = str.substr(0, colon);
    int index = 0;
    if (colon != std::string_view::npos) {
        std::string idx_str(str.substr(colon + 1));
        try {
            index = std::stoi(idx_str);
            if (index < 0) return std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }

    std::string lower_base;
    lower_base.reserve(base.size());
    for (char c : base) {
        lower_base.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower_base == "cpu") return Device(DeviceType::CPU, index);
    if (lower_base == "ispc") return Device(DeviceType::ISPC, index);
    if (lower_base == "cuda") return Device(DeviceType::CUDA, index);
    if (lower_base == "vulkan") return Device(DeviceType::VULKAN, index);

    return std::nullopt;
}

auto Device::available_devices() -> std::vector<Device> {
    std::vector<Device> res;
    if (Device::cpu().is_available()) {
        res.push_back(Device::cpu());
    }
    if (Device::ispc().is_available()) {
        res.push_back(Device::ispc());
    }
#if defined(VELOMIND_ENABLE_CUDA)
    int cuda_count = 0;
    if (Device::cuda(0).is_available() && cudaGetDeviceCount(&cuda_count) == cudaSuccess) {
        for (int i = 0; i < cuda_count; ++i) {
            res.push_back(Device::cuda(i));
        }
    } else if (Device::cuda(0).is_available()) {
        res.push_back(Device::cuda(0));
    }
#else
    if (Device::cuda(0).is_available()) {
        res.push_back(Device::cuda(0));
    }
#endif
    if (Device::vulkan(0).is_available()) {
        res.push_back(Device::vulkan(0));
    }
    return res;
}

auto Device::default_device() -> Device {
    if (Device::cuda(0).is_available()) return Device::cuda(0);
    if (Device::vulkan(0).is_available()) return Device::vulkan(0);
    if (Device::ispc().is_available()) return Device::ispc();
    return Device::cpu();
}

auto Device::is_available() const noexcept -> bool {
    return internal::is_device_available(_type);
}

auto Device::name() const -> std::string {
    std::string base;
    switch (_type) {
        case DeviceType::CPU:    base = "CPU"; break;
        case DeviceType::ISPC:   base = "ISPC"; break;
        case DeviceType::CUDA:   base = "CUDA"; break;
        case DeviceType::VULKAN: base = "Vulkan"; break;
        default:                 base = "Unknown"; break;
    }
    if (_index > 0) {
        base += ":" + std::to_string(_index);
    }
    return base;
}

auto Device::description() const -> std::string {
    if (_type == DeviceType::CPU) {
        std::ifstream cpuinfo("/proc/cpuinfo");
        if (cpuinfo) {
            std::string line;
            while (std::getline(cpuinfo, line)) {
                if (line.rfind("model name", 0) == 0) {
                    auto colon = line.find(':');
                    if (colon != std::string::npos && colon + 2 < line.size()) {
                        return line.substr(colon + 2);
                    }
                }
            }
        }
        return "Host CPU";
    }
    if (_type == DeviceType::ISPC) {
        return "ISPC SIMD Vectorized Host Runtime";
    }
#if defined(VELOMIND_ENABLE_CUDA)
    if (_type == DeviceType::CUDA) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && _index < device_count) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, _index) == cudaSuccess) {
                return prop.name;
            }
        }
        return "NVIDIA CUDA Device";
    }
#endif
#if defined(VELOMIND_ENABLE_VULKAN)
    if (_type == DeviceType::VULKAN) {
        return "Vulkan Compute Device";
    }
#endif
    return "Unknown Device";
}

auto Device::supports(DataType dtype) const noexcept -> bool {
    if (!is_available()) return false;
    return is_dtype_supported(_type, dtype);
}

auto Device::supports(
    Op op,
    DataType dtype
) const noexcept -> bool {
    std::array<DataType, 1> in1{dtype};
    if (supports(op, in1, dtype)) return true;
    std::array<DataType, 2> in2{dtype, dtype};
    return supports(op, in2, dtype);
}

auto Device::supports(
    Op op,
    std::span<const DataType> input_dtypes,
    DataType output_dtype
) const noexcept -> bool {
    if (!is_available()) return false;
    return is_op_dtype_supported(_type, op, input_dtypes, output_dtype);
}

auto Device::memory_info() const -> std::optional<DeviceMemoryInfo> {
    if (!is_available()) return std::nullopt;

#if defined(VELOMIND_ENABLE_CUDA)
    if (_type == DeviceType::CUDA) {
        int cur_dev = 0;
        cudaGetDevice(&cur_dev);
        if (cur_dev != _index) {
            cudaSetDevice(_index);
        }
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            if (cur_dev != _index) {
                cudaSetDevice(cur_dev);
            }
            return DeviceMemoryInfo{
                .total_bytes = total_b,
                .used_bytes  = total_b - free_b,
                .free_bytes  = free_b
            };
        }
        if (cur_dev != _index) {
            cudaSetDevice(cur_dev);
        }
    }
#endif

    if (_type == DeviceType::CPU || _type == DeviceType::ISPC) {
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo) {
            std::string line;
            std::size_t total_kb = 0, free_kb = 0, avail_kb = 0;
            while (std::getline(meminfo, line)) {
                if (line.rfind("MemTotal:", 0) == 0) {
                    std::istringstream iss(line.substr(9));
                    iss >> total_kb;
                } else if (line.rfind("MemAvailable:", 0) == 0) {
                    std::istringstream iss(line.substr(13));
                    iss >> avail_kb;
                } else if (line.rfind("MemFree:", 0) == 0) {
                    std::istringstream iss(line.substr(8));
                    iss >> free_kb;
                }
            }
            if (total_kb > 0) {
                std::size_t eff_free = (avail_kb > 0 ? avail_kb : free_kb) * 1024;
                std::size_t total = total_kb * 1024;
                return DeviceMemoryInfo{
                    .total_bytes = total,
                    .used_bytes  = total > eff_free ? total - eff_free : 0,
                    .free_bytes  = eff_free
                };
            }
        }
    }

    return std::nullopt;
}

auto to_string(const Device& device) -> std::string {
    return device.name();
}

auto operator<<(std::ostream& os, const Device& device) -> std::ostream& {
    return os << device.name();
}

} // namespace velomind
