#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "velomind/node.h"
#include "velomind/ops.h"
#include "velomind/ptr.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind_core_export.h"

namespace velomind {

class Executable;

template <typename T>
concept TensorHandle = std::is_same_v<std::remove_cvref_t<T>, Tensor>;

template <typename... Ts>
concept AllTensorHandles = (TensorHandle<Ts> && ...);

// 默认初始预留容量：兼顾单测、微基准以及小型计算图的内存脚印与分配延迟。
inline constexpr std::size_t DEFAULT_TENSOR_CAPACITY = 64;
inline constexpr std::size_t DEFAULT_NODE_CAPACITY   = 32;

// 历史保留建议容量；明确表示该值并非系统硬上限，Graph 容器随构图需求动态自动扩容。
inline constexpr std::size_t MAX_TENSORS_PER_GRAPH = 1U << 16;

// 内存规划与中间值复用统计指标。
struct MemoryPlanStats {
    std::size_t peak_bytes = 0;                 // 规划后的物理内存峰值（字节）
    std::size_t allocation_count = 0;           // 实际发起的底层物理分配次数
    std::size_t intermediate_tensor_count = 0;  // 参与复用规划的中间临时张量总数
    std::size_t reused_tensor_count = 0;        // 成功共享复用内存的张量数
    std::size_t bytes_saved = 0;                // 相比独立分配节省的内存字节数
};

// Graph 持有全部张量存储与计算节点，定义有向无环计算图（DAG）。
// 构图期间仅推导形状签名与拓扑边，不触发硬件分配；由它产生的 Tensor 句柄与编译出的 Executable 执行计划均依赖其生命周期。
class VELOMIND_CORE_EXPORT Graph {
public:
    // 构造计算图；支持调用方指定张量与节点初始容量提示（默认 64 张量、32 节点）。
    // 容量提示仅用于预热内部容器以消除构图扩容开销，Graph 具备完全动态扩容能力，不受预留上限约束。
    explicit Graph(std::size_t initial_tensor_capacity = DEFAULT_TENSOR_CAPACITY,
                   std::size_t initial_node_capacity   = DEFAULT_NODE_CAPACITY);
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;

    // 显式调整张量与节点预留容量；图构建完成后调用将抛出异常。
    auto reserve(std::size_t tensor_capacity, std::size_t node_capacity = 0) -> void;

    auto tensor_capacity() const noexcept -> std::size_t { return _tensors.capacity(); }
    auto node_capacity()   const noexcept -> std::size_t { return _nodes.capacity(); }

    // 声明图的外部输入张量占位符（此时仅记录形状与类型，不分配真实存储；在 build 阶段规划或由 copy_from_host 填充）。
    auto input(shape_t shape, DataType dtype) -> Tensor;

    // 接入外部已有存储作为图输入；常用于跨会话推理或多个子图共享同一套模型权重（零拷贝）。
    auto input(std::shared_ptr<TensorStorage> storage) -> Tensor;

    auto tensor_storage_at(std::size_t i) const -> pConstTensorStorage;
    auto shared_storage_at(std::size_t i) const -> std::shared_ptr<TensorStorage>;
    auto tensor_storage_at(std::size_t i)       -> pTensorStorage;

    auto tensor_count() const noexcept -> std::size_t { return _tensors.size(); }
    auto is_input_tensor(std::size_t i) const noexcept -> bool;
    auto is_output_tensor(std::size_t i) const noexcept -> bool;
    auto lifetime_token() const noexcept -> std::weak_ptr<void> { return _lifetime; }

    // 在图中追加算子节点：根据全局算子签名注册表（OpSignature）自动推导输出张量的形状与数据类型。
    template <typename... Inputs>
        requires AllTensorHandles<Inputs...>
    auto op(Op op, Inputs&&... inputs) -> Tensor;

    template <typename... Inputs>
        requires AllTensorHandles<Inputs...>
    auto op(OpDescriptor desc, Inputs&&... inputs) -> Tensor;

    // 显式指定图的终端输出张量集合。未被显式输出依赖的游离节点将在 build 编译阶段被死代码消除（DCE）剪枝。
    auto mark_output(Tensor tensor) -> void;
    auto set_outputs(std::vector<Tensor> outputs) -> void;
    auto outputs() const -> std::vector<Tensor>;
    auto has_explicit_outputs() const noexcept -> bool { return _has_explicit_outputs; }

    // 静态校验图结构：检查 DAG 有向无环性、算子形状契约与目标后端内核注册可用性，不触发任何内存分配。
    auto validate(DeviceType device) const -> void;

    // 编译计算图为可执行计划（Executable）：
    // 执行 DCE 死代码消除、拓扑排序、基于张量生命周期的内存规划（复用中间缓冲区）并在指定设备上物理分配内存。
    auto build(DeviceType device) -> std::unique_ptr<Executable>;

    auto is_built() const noexcept -> bool { return _built; }
    auto bound_device() const noexcept -> std::optional<DeviceType> {
        return _built ? std::make_optional(_bound_device) : std::nullopt;
    }

    auto memory_plan_stats() const noexcept -> const std::optional<MemoryPlanStats>& {
        return _plan_stats;
    }

    auto node_count() const noexcept -> std::size_t { return _nodes.size(); }
    auto node_tot() const -> std::size_t { return _nodes.size(); }

private:
    auto _op_impl(OpDescriptor desc, std::vector<pConstTensorStorage> inputs) -> Tensor;

    auto _compute_required_nodes() const -> std::vector<bool>;
    auto _build_producer_map() const -> std::unordered_map<pConstTensorStorage, pConstNode>;
    auto _topological_order()  const -> std::vector<pConstNode>;

    friend struct GraphTestAccess;

    std::vector<std::shared_ptr<TensorStorage>> _tensors;
    std::vector<Node>                           _nodes;
    bool                                        _built = false;
    DeviceType                                  _bound_device = DeviceType::CPU;
    bool                                        _has_explicit_outputs = false;
    std::vector<pConstTensorStorage>            _explicit_outputs;
    std::optional<MemoryPlanStats>              _plan_stats;
    std::shared_ptr<void>                       _lifetime;
};

template <typename... Inputs>
    requires AllTensorHandles<Inputs...>
auto Graph::op(Op op_kind, Inputs&&... inputs) -> Tensor {
    return op(OpDescriptor{ op_kind, NoAttrs{} }, std::forward<Inputs>(inputs)...);
}

template <typename... Inputs>
    requires AllTensorHandles<Inputs...>
auto Graph::op(OpDescriptor desc, Inputs&&... inputs) -> Tensor {
    if (_built) throw std::logic_error("Graph::op: graph is already built");
    if (((inputs.graph() != this || inputs.storage() == nullptr) || ...)) {
        throw std::invalid_argument("Graph::op: invalid or foreign tensor handle");
    }
    std::vector<pConstTensorStorage> ptrs;
    ptrs.reserve(sizeof...(inputs));
    ((ptrs.push_back(inputs.storage())), ...);
    return _op_impl(std::move(desc), std::move(ptrs));
}

VELOMIND_EXPORT_PTR(Graph);

} // namespace velomind
