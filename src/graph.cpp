#include "velomind/graph.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"

#include "internal/graph_compiler.h"
#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/op_validation.h"

namespace velomind {

auto Graph::_build_producer_map() const -> std::unordered_map<pConstTensorStorage, pConstNode> {
    std::unordered_map<pConstTensorStorage, pConstNode> producer;
    producer.reserve(_nodes.size());
    for (const auto& n : _nodes) {
        for (auto* out : n.outputs()) {
            producer[out] = &n;
        }
    }
    return producer;
}

auto Graph::_compute_required_nodes() const -> std::vector<bool> {
    if (!_has_explicit_outputs) {
        return std::vector<bool>(_nodes.size(), true);
    }

    std::unordered_map<pConstTensorStorage, std::size_t> producer_map;
    producer_map.reserve(_nodes.size());
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
        for (auto* out : _nodes[i].outputs()) {
            producer_map[out] = i;
        }
    }

    std::vector<bool> required(_nodes.size(), false);
    std::queue<std::size_t> worklist;

    for (auto* out_storage : _explicit_outputs) {
        auto it = producer_map.find(out_storage);
        if (it != producer_map.end()) {
            if (!required[it->second]) {
                required[it->second] = true;
                worklist.push(it->second);
            }
        }
    }

    while (!worklist.empty()) {
        std::size_t u = worklist.front();
        worklist.pop();
        for (auto* in_storage : _nodes[u].inputs()) {
            auto it = producer_map.find(in_storage);
            if (it != producer_map.end()) {
                if (!required[it->second]) {
                    required[it->second] = true;
                    worklist.push(it->second);
                }
            }
        }
    }

    return required;
}

auto Graph::_topological_order() const -> std::vector<pConstNode> {
    if (_nodes.empty()) return {};

    const auto required = _compute_required_nodes();
    std::size_t active_count = 0;
    for (bool req : required) {
        if (req) ++active_count;
    }
    if (active_count == 0) return {};

    // 建立输出存储到生产者节点索引的映射。
    std::unordered_map<pConstTensorStorage, std::size_t> producer_map;
    producer_map.reserve(_nodes.size());
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
        for (auto* out : _nodes[i].outputs()) {
            producer_map[out] = i;
        }
    }

    // 建立活跃节点的消费者邻接表与去重后的前置依赖入度。
    std::vector<std::size_t> in_degree(_nodes.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(_nodes.size());

    std::vector<std::size_t> unique_producers;
    for (std::size_t v = 0; v < _nodes.size(); ++v) {
        if (!required[v]) continue;
        unique_producers.clear();
        for (auto* in : _nodes[v].inputs()) {
            auto it = producer_map.find(in);
            if (it != producer_map.end() && required[it->second]) {
                unique_producers.push_back(it->second);
            }
        }
        std::sort(unique_producers.begin(), unique_producers.end());
        unique_producers.erase(
            std::unique(unique_producers.begin(), unique_producers.end()),
            unique_producers.end());

        in_degree[v] = unique_producers.size();
        for (std::size_t u : unique_producers) {
            consumers[u].push_back(v);
        }
    }

    // Kahn 算法：按零入度驱动拓扑排序。
    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
        if (required[i] && in_degree[i] == 0) ready.push(i);
    }

    std::vector<pConstNode> order;
    order.reserve(active_count);

    while (!ready.empty()) {
        std::size_t u = ready.front();
        ready.pop();
        order.push_back(&_nodes[u]);
        for (std::size_t v : consumers[u]) {
            if (--in_degree[v] == 0) {
                ready.push(v);
            }
        }
    }

    // 存在有向环或依赖不完整时拒绝构建，防止未就绪节点被执行。
    if (order.size() != active_count) {
        throw std::invalid_argument("Graph::build: cycle detected in graph");
    }

    return order;
}

Graph::Graph(std::size_t initial_tensor_capacity, std::size_t initial_node_capacity)
    : _lifetime(std::make_shared<int>(0)) {
    if (initial_tensor_capacity > 0) {
        _tensors.reserve(initial_tensor_capacity);
    }
    if (initial_node_capacity > 0) {
        _nodes.reserve(initial_node_capacity);
    }
}

Graph::~Graph() = default;

auto Graph::reserve(std::size_t tensor_capacity, std::size_t node_capacity) -> void {
    if (_built) {
        throw std::logic_error("Graph::reserve: graph is already built");
    }
    if (tensor_capacity > _tensors.capacity()) {
        _tensors.reserve(tensor_capacity);
    }
    if (node_capacity > _nodes.capacity()) {
        _nodes.reserve(node_capacity);
    }
}

auto Graph::is_input_tensor(std::size_t i) const noexcept -> bool {
    auto* storage = tensor_storage_at(i);
    if (!storage) return false;
    for (const auto& node : _nodes)
        for (auto* output : node.outputs())
            if (output == storage) return false;
    return true;
}

auto Graph::is_output_tensor(std::size_t i) const noexcept -> bool {
    auto* storage = tensor_storage_at(i);
    if (!storage) return false;
    if (_has_explicit_outputs) {
        for (auto* out : _explicit_outputs) {
            if (out == storage) return true;
        }
        return false;
    }
    if (is_input_tensor(i)) return false;
    for (const auto& node : _nodes) {
        for (auto* input : node.inputs()) {
            if (input == storage) return false;
        }
    }
    return true;
}

auto Graph::tensor_storage_at(std::size_t i) const -> pConstTensorStorage {
    return i < _tensors.size() ? _tensors[i].get() : nullptr;
}

auto Graph::tensor_storage_at(std::size_t i) -> pTensorStorage {
    return i < _tensors.size() ? _tensors[i].get() : nullptr;
}

auto Graph::shared_storage_at(std::size_t i) const -> std::shared_ptr<TensorStorage> {
    return i < _tensors.size() ? _tensors[i] : nullptr;
}

auto Graph::input(shape_t shape, DataType dtype) -> Tensor {
    if (_built) throw std::logic_error("Graph::input: graph is already built");
    auto storage = std::make_shared<TensorStorage>();
    storage->shape          = std::move(shape);
    storage->strides        = default_strides(storage->shape);
    storage->dtype          = dtype;
    storage->device         = DeviceType::CPU;
    storage->size_bytes     = storage_nbytes(*storage);
    storage->capacity_bytes = storage->size_bytes;
    storage->offset_bytes   = 0;

    _tensors.push_back(std::move(storage));
    return Tensor{this, _tensors.size() - 1};
}

auto Graph::input(std::shared_ptr<TensorStorage> storage) -> Tensor {
    if (_built) throw std::logic_error("Graph::input: graph is already built");
    if (!storage) {
        throw std::invalid_argument("Graph::input: null storage");
    }
    if (storage->size_bytes < storage_nbytes(*storage)) {
        throw std::invalid_argument("Graph::input: storage capacity is smaller than tensor");
    }
    if (storage->strides.empty()) {
        storage->strides = default_strides(storage->shape);
    }
    if (storage->capacity_bytes == 0) {
        storage->capacity_bytes = storage->size_bytes;
    }
    _tensors.push_back(std::move(storage));
    return Tensor{this, _tensors.size() - 1};
}

auto Graph::_op_impl(OpDescriptor desc, std::vector<pConstTensorStorage> inputs) -> Tensor {
    Node node;
    Op op_kind = desc.op;
    if (std::holds_alternative<NoAttrs>(desc.attrs)) {
        desc.attrs = internal::default_op_attrs(op_kind);
    }
    OpAttrs attrs = desc.attrs;
    node._op = std::move(desc);
    node._inputs = std::move(inputs);

    // 形状与类型统一由算子签名推导，避免内核注册规则和图元数据分叉。
    auto signature = internal::op_signature(
        op_kind,
        std::span<const pConstTensorStorage>(node._inputs.data(), node._inputs.size()),
        attrs);

    auto out_storage = std::make_shared<TensorStorage>();
    out_storage->shape          = std::move(signature.shape);
    out_storage->strides        = default_strides(out_storage->shape);
    out_storage->dtype          = signature.dtype;
    out_storage->device         = DeviceType::CPU;
    out_storage->size_bytes     = storage_nbytes(*out_storage);
    out_storage->capacity_bytes = out_storage->size_bytes;
    out_storage->offset_bytes   = 0;

    _tensors.push_back(std::move(out_storage));
    std::size_t idx = _tensors.size() - 1;
    node._outputs = {pTensorStorage{_tensors[idx].get()}};

    _nodes.push_back(std::move(node));
    return Tensor{this, idx};
}

auto Graph::mark_output(Tensor tensor) -> void {
    if (_built) throw std::logic_error("Graph::mark_output: graph is already built");
    if (tensor.graph() != this || tensor.storage() == nullptr) {
        throw std::invalid_argument("Graph::mark_output: invalid or foreign tensor handle");
    }
    _has_explicit_outputs = true;
    for (auto* s : _explicit_outputs) {
        if (s == tensor.storage()) return;
    }
    _explicit_outputs.push_back(tensor.storage());
}

auto Graph::set_outputs(std::vector<Tensor> outputs) -> void {
    if (_built) throw std::logic_error("Graph::set_outputs: graph is already built");
    _explicit_outputs.clear();
    if (outputs.empty()) {
        _has_explicit_outputs = false;
        return;
    }
    _has_explicit_outputs = true;
    for (const auto& t : outputs) {
        if (t.graph() != this || t.storage() == nullptr) {
            throw std::invalid_argument("Graph::set_outputs: invalid or foreign tensor handle");
        }
        bool exists = false;
        for (auto* s : _explicit_outputs) {
            if (s == t.storage()) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            _explicit_outputs.push_back(t.storage());
        }
    }
}

auto Graph::outputs() const -> std::vector<Tensor> {
    std::vector<Tensor> result;
    if (_has_explicit_outputs) {
        result.reserve(_explicit_outputs.size());
        for (auto* s : _explicit_outputs) {
            for (std::size_t i = 0; i < _tensors.size(); ++i) {
                if (_tensors[i].get() == s) {
                    result.push_back(Tensor{const_cast<Graph*>(this), i});
                    break;
                }
            }
        }
    } else {
        for (std::size_t i = 0; i < _tensors.size(); ++i) {
            if (is_output_tensor(i)) {
                result.push_back(Tensor{const_cast<Graph*>(this), i});
            }
        }
    }
    return result;
}

auto Graph::validate(DeviceType device) const -> void {
    if (_built) {
        throw std::logic_error("Graph::validate: graph is already built");
    }

    if (static_cast<std::size_t>(device) >= MAX_DEVICE_TYPES) {
        throw std::invalid_argument("Graph::validate: invalid device");
    }

    for (const auto& storage : _tensors) {
        if (storage->size_bytes < storage_nbytes(*storage)) {
            throw std::invalid_argument("Graph::validate: invalid tensor capacity");
        }
        if (storage->data != nullptr && storage->device != device) {
            throw std::invalid_argument("Graph::validate: populated storage belongs to another device");
        }
    }

    const auto required = _compute_required_nodes();

    // 元数据可经共享存储修改；重新验证活跃节点签名与内核可用性。
    for (std::size_t i = 0; i < _nodes.size(); ++i) {
        if (!required[i]) continue;
        const auto& node = _nodes[i];
        if (node.outputs().empty()) {
            throw std::invalid_argument("Graph::validate: node has no outputs");
        }
        const auto signature = internal::op_signature(node.op().op, node.inputs(), node.op().attrs);
        const auto* output = node.outputs().front();
        if (signature.shape != output->shape || signature.dtype != output->dtype) {
            throw std::invalid_argument("Graph::validate: output metadata no longer matches signature");
        }
        internal::validate_backend_index_range(node.op().op, node.inputs(), *output, device);
        std::vector<DataType> dtypes;
        dtypes.reserve(node.inputs().size());
        for (const auto* input : node.inputs()) {
            dtypes.push_back(input->dtype);
        }
        if (!internal::resolve_op_kernel(node.op().op, dtypes, output->dtype, device)) {
            throw std::invalid_argument(std::string("Graph::validate: unsupported kernel for ") + op_name(node.op().op));
        }
    }

    // 拓扑排序阶段检查 DAG 是否存在环路；不触发任何设备内存分配。
    static_cast<void>(_topological_order());
}

auto Graph::build(DeviceType device) -> std::unique_ptr<Executable> {
    return internal::GraphCompiler::compile(*this, device);
}

} // namespace velomind

