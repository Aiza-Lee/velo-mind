#pragma once

#include <cstddef>
#include <memory>

#include "velomind/graph.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind_core_export.h"

namespace velomind {

class Graph;

// 封装外部预分配缓冲区的描述符；通过 owner 保持对外部生命周期的引用，支持在图输入端实现零拷贝接入。
struct ExternalBuffer {
    void*       data           = nullptr;
    std::size_t capacity_bytes = 0;
    DeviceType  device         = DeviceType::CPU;
    std::shared_ptr<void> owner;
};

// 已完成编译与内核解析的计算图执行计划；所属 Graph 必须覆盖其生命周期。
// 维护顺序编译节点拓扑队列（CompiledNode）与设备特化上下文（如 CUDA 异步流与 cuBLAS 句柄）。
class VELOMIND_CORE_EXPORT Executable {
public:

    using KernelFn = ::velomind::KernelFn;

    // 编译期算子执行单元：绑定经解析的具体硬件核函数、输入输出物理存储指针及强类型属性
    struct CompiledNode {
        KernelFn                          kernel = nullptr;
        std::vector<pConstTensorStorage>  inputs;
        std::vector<pTensorStorage>       outputs;
        std::shared_ptr<const void>       attrs;
        std::vector<TensorStorage>        input_metadata;
        std::vector<TensorStorage>        output_metadata;
    };

    struct Impl {
        pConstGraph               graph  = nullptr;
        std::weak_ptr<void>       graph_lifetime;
        DeviceType                device = DeviceType::CPU;
        std::vector<CompiledNode> topo;
        MemoryPlanStats           plan_stats;
        // 后端特化执行上下文（如 CUDA 流与 cuBLAS 句柄），生命周期随 Executable 析构释放
        std::shared_ptr<void>     device_context;
    };

    Executable() = default;
    ~Executable();

    Executable(const Executable&) = delete;
    Executable& operator=(const Executable&) = delete;

    Executable(Executable&&) noexcept;
    Executable& operator=(Executable&&) noexcept;

    // 按拓扑顺序触发执行所有编译节点；对于 GPU 后端在对应流上提交异步核函数并在边界进行同步管理。
    auto execute() -> void;

    // 将外部缓冲区绑定到指定图输入张量；owner 保证外部物理存储在执行期间存活。
    auto bind_input(Tensor tensor, ExternalBuffer external_buffer) -> void;

    // 查询指定输出张量的底层存储指针，用于读取推理结果。
    auto output(Tensor tensor)       -> pTensorStorage;
    auto output(Tensor tensor) const -> pConstTensorStorage;

    auto memory_plan_stats() const noexcept -> const MemoryPlanStats&;

    auto num_nodes() const -> std::size_t;
    auto empty()     const -> bool { return num_nodes() == 0; }

private:
    friend class Graph;
    std::unique_ptr<Impl> _impl;
};

} // namespace velomind
