#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

#include "velomind/ptr.h"
#include "velomind/types.h"

namespace velomind {

// 计算行优先连续（C-contiguous）布局的默认元素步长。
inline auto default_strides(const shape_t& shape) -> stride_t {
    if (shape.empty()) return {};
    stride_t s(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;) {
        dim_t dim = std::max<dim_t>(0, shape[i]);
        dim_t prod = 0;
        if (__builtin_mul_overflow(s[i], dim, &prod)) {
            prod = std::numeric_limits<dim_t>::max();
        }
        s[i - 1] = prod;
    }
    return s;
}

// 检查给定形状与步长是否满足行优先连续布局。
inline auto is_shape_contiguous(const shape_t& shape, const stride_t& strides) -> bool {
    if (strides.empty() || shape.empty()) return true;
    if (strides.size() != shape.size()) return false;
    dim_t expected_stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (shape[i] <= 0) return true;
        if (shape[i] == 1) continue;
        if (strides[i] != expected_stride) return false;
        dim_t next_stride = 0;
        if (__builtin_mul_overflow(expected_stride, shape[i], &next_stride)) {
            return false;
        }
        expected_stride = next_stride;
    }
    return true;
}

// 后端分配器返回的物理存储或视图别名描述；定义内存布局契约、步长规则与生命周期托管。
// 既可代表拥有实际物理分配内存的主存储，也可通过 external_owner 引用底层缓冲区以表达切片、转置等零拷贝视图。
struct TensorStorage {

    shape_t     shape;
    stride_t    strides;          // 元素步长；为空时默认遵循行优先连续步长
    std::size_t offset_bytes   = 0; // 视图在底层物理缓冲区中的起始字节偏移
    std::size_t capacity_bytes = 0; // 底层物理分配的总字节容量（支持 Arena 内存池超额复用）
    DataType    dtype          = DataType::Float32;
    DeviceType  device         = DeviceType::CPU;
    void*       data           = nullptr; // 硬件缓冲区物理首地址（主机端或设备显存）
    std::size_t size_bytes     = 0; // 本张量逻辑占用的连续字节数
    std::shared_ptr<void> external_owner; // 若本存储为别名视图，持有底层缓冲区强引用以防止悬垂指针

    // 依据设备类型分发调用对应后端注册的 StorageCreator 分配指定容量的物理内存
    static auto allocate(std::size_t bytes, DeviceType device) -> std::shared_ptr<TensorStorage>;

    // 查询目标设备分配器是否已成功注册并可用
    static auto is_available(DeviceType device) noexcept -> bool;

    auto effective_strides() const -> stride_t {
        if (strides.empty()) return default_strides(shape);
        return strides;
    }

    auto is_contiguous() const -> bool {
        return is_shape_contiguous(shape, strides);
    }

    auto is_alias() const noexcept -> bool {
        return external_owner != nullptr;
    }
};

inline auto storage_numel(const TensorStorage& s) -> std::size_t {
    bool empty = false;
    for (auto d : s.shape) {
        if (d < 0) throw std::invalid_argument("Tensor: negative dimension");
        empty = empty || d == 0;
    }
    if (empty) return 0;
    std::size_t n = 1;
    for (auto d : s.shape) {
        if (static_cast<std::uint64_t>(d) > std::numeric_limits<std::size_t>::max() / n)
            throw std::invalid_argument("Tensor: element count overflow");
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

inline auto storage_nbytes(const TensorStorage& s) -> std::size_t {
    const auto width = data_type_size(s.dtype);
    if (width == 0) throw std::invalid_argument("Tensor: invalid dtype");
    const auto n = storage_numel(s);
    if (n > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("Tensor: byte count overflow");
    return n * width;
}

// 格式化物理存储信息为可读字符串
inline auto to_string(const TensorStorage& s) -> std::string {
    std::string res = "TensorStorage(shape=" + to_string(s.shape);
    res += ", dtype=" + std::string(data_type_name(s.dtype));
    res += ", device=" + std::string(device_type_name(s.device));
    if (!s.strides.empty()) {
        res += ", strides=" + to_string(s.strides);
    }
    if (s.offset_bytes > 0) {
        res += ", offset_bytes=" + std::to_string(s.offset_bytes);
    }
    if (s.is_alias()) {
        res += ", alias=true";
    }
    res += (s.data != nullptr ? ", allocated" : ", unallocated");
    res += ")";
    return res;
}

inline auto operator<<(std::ostream& os, const TensorStorage& s) -> std::ostream& {
    return os << to_string(s);
}

VELOMIND_EXPORT_PTR(TensorStorage);

} // namespace velomind
