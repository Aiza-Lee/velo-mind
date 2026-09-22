#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

#include "velomind/types.h"

#ifdef VELOMIND_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace velomind::benchmark {

struct ProcessMemory {
    double vm_rss_mb = 0.0;
    double vm_hwm_mb = 0.0;
    double vm_peak_mb = 0.0;
};

// 从 /proc/self/status 读取进程物理内存常驻集 (RSS) 与历史峰值 (HWM)。
inline auto get_process_memory() -> ProcessMemory {
    ProcessMemory mem;
    std::ifstream status_file("/proc/self/status");
    if (!status_file) {
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            mem.vm_hwm_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
            mem.vm_rss_mb = mem.vm_hwm_mb;
        }
        return mem;
    }

    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            if (iss >> kb) mem.vm_rss_mb = kb / 1024.0;
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            if (iss >> kb) mem.vm_hwm_mb = kb / 1024.0;
        } else if (line.rfind("VmPeak:", 0) == 0) {
            std::istringstream iss(line.substr(7));
            double kb = 0.0;
            if (iss >> kb) mem.vm_peak_mb = kb / 1024.0;
        }
    }
    return mem;
}

struct GpuMemory {
    double total_mb = 0.0;
    double free_mb  = 0.0;
    double used_mb  = 0.0;
};

// 查询当前活动 CUDA 设备显存分配状态。
inline auto get_cuda_memory() -> GpuMemory {
    GpuMemory gpu;
#ifdef VELOMIND_ENABLE_CUDA
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        gpu.total_mb = static_cast<double>(total_b) / (1024.0 * 1024.0);
        gpu.free_mb  = static_cast<double>(free_b) / (1024.0 * 1024.0);
        gpu.used_mb  = gpu.total_mb - gpu.free_mb;
    }
#endif
    return gpu;
}

struct SystemMetadata {
    std::string cpu_model;
    std::size_t cpu_concurrency = 0;
    double      ram_total_gb    = 0.0;
    std::string gpu_model;
    double      gpu_vram_gb     = 0.0;
    std::string os_version;
    std::string compiler;
    std::string build_type;
};

inline auto get_system_metadata() -> SystemMetadata {
    SystemMetadata meta;
    meta.cpu_concurrency = std::thread::hardware_concurrency();

    // 读取 CPU 型号
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("model name", 0) == 0) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    meta.cpu_model = line.substr(colon + 2);
                    break;
                }
            }
        }
    }

    // 读取物理总内存
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.rfind("MemTotal:", 0) == 0) {
                std::istringstream iss(line.substr(9));
                double kb = 0.0;
                if (iss >> kb) meta.ram_total_gb = kb / (1024.0 * 1024.0);
                break;
            }
        }
    }

    // 读取操作系统内核
    std::ifstream osinfo("/proc/sys/kernel/osrelease");
    if (osinfo) {
        std::getline(osinfo, meta.os_version);
    }

    // 读取 GPU 型号与总显存
#ifdef VELOMIND_ENABLE_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            meta.gpu_model = prop.name;
            meta.gpu_vram_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        }
    }
#endif

#if defined(__clang__)
    meta.compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    meta.compiler = std::string("GCC ") + __VERSION__;
#else
    meta.compiler = "Unknown";
#endif

#if defined(NDEBUG)
    meta.build_type = "Release";
#else
    meta.build_type = "Debug";
#endif

    return meta;
}

} // namespace velomind::benchmark
