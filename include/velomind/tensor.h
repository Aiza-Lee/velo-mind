#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "velomind/ptr.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind_core_export.h"

namespace velomind {

class Graph;

// Tensor 是指向 Graph 内部存储的轻量句柄，不直接持有物理内存所有权。
// 通过图索引绑定至 Graph 统一管理的 TensorStorage，受弱引用保护以防悬垂；
// 支持表达行优先连续数据、跨步视图（如转置）以及共享底层内存的切片别名。
class VELOMIND_CORE_EXPORT Tensor {
public:

    Tensor() = default;

    Tensor(Graph* graph, std::size_t index) noexcept;

    Tensor(const Tensor&)                = default;
    Tensor& operator=(const Tensor&)     = default;
    Tensor(Tensor&&) noexcept            = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    auto shape()  const -> const shape_t&;
    auto dtype()  const -> DataType;
    auto device() const -> DeviceType;
    auto numel()  const -> std::size_t;
    auto nbytes() const -> std::size_t;

    // 跨步视图元数据：支持非连续步长、切片偏移与底层分配容量查询
    auto strides()        const -> stride_t;
    auto offset_bytes()   const -> std::size_t;
    auto capacity_bytes() const -> std::size_t;
    auto is_contiguous()  const -> bool;
    auto is_alias()       const -> bool;

    // 返回底层缓冲区的首地址（已叠加 offset_bytes 视图偏移）。
    // 若张量位于 CUDA/Vulkan 等非 CPU 设备，返回指针为设备显存地址，主机端不可直接解引用。
    auto data() const -> const void*;
    auto data()       -> void*;

    // 内部存储原始指针访问；生命周期严格受所属 Graph 约束
    auto storage()     const -> pConstTensorStorage;
    auto storage()           -> pTensorStorage;

    // 显式提取底层存储的 std::shared_ptr 共享所有权。
    // 用于在不同 Graph 实例间安全共享权重模型（如多会话推理）或脱离 Graph 生命周期持久化数据。
    auto shared_storage() const -> std::shared_ptr<TensorStorage>;

    auto has_storage() const -> bool { return storage() != nullptr; }

    auto index() const noexcept -> std::size_t { return _index; }

    auto graph() const noexcept -> Graph* { return _graph; }

    explicit operator bool() const noexcept { return storage() != nullptr; }

    // 主机与设备间数据搬运接口：自动解析目标设备的 MemoryTransfer 驱动。
    // 带 offset 重载支持切片局部回传（如自回归解码中仅将末行 Logits 回传至 Host，避免全序列冗余传输）。
    auto copy_from_host(std::span<const std::byte> data)                              -> void;
    auto copy_from_host(std::span<const std::byte> data, std::size_t dst_offset_bytes) -> void;
    auto copy_to_host(std::span<std::byte> data)                                const -> void;
    auto copy_to_host(std::span<std::byte> data, std::size_t src_offset_bytes)        const -> void;

    // 转换为人类可读的字符串描述；print_data 为 true 时若张量位于 CPU 且已分配则输出完整内容预览
    auto to_string(bool print_data = false) const -> std::string;

    friend auto operator==(const Tensor& a, const Tensor& b) noexcept -> bool {
        return a._graph == b._graph && a._index == b._index;
    }
    friend auto operator!=(const Tensor& a, const Tensor& b) noexcept -> bool {
        return !(a == b);
    }

private:
    friend class Graph;

    Graph*       _graph = nullptr;
    std::size_t  _index = 0;
    std::weak_ptr<void> _graph_lifetime;
};

inline auto to_string(const Tensor& t, bool print_data = false) -> std::string {
    return t.to_string(print_data);
}

inline auto operator<<(std::ostream& os, const Tensor& t) -> std::ostream& {
    return os << t.to_string();
}

VELOMIND_EXPORT_PTR(Tensor);

} // namespace velomind
