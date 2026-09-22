#include "velomind/graph.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#ifdef VELOMIND_ENABLE_CUDA
#include "cuda/cuda_context.h"
#endif

#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/registry/storage.h"
#include "internal/op_validation.h"

namespace velomind {

namespace {

    // 扩展状态与 TensorStorage 内存连续布局，供 Vulkan 算子反查 VkBuffer。
    struct VulkanExt {
        VkBuffer        buffer   = VK_NULL_HANDLE;
        VkDeviceMemory  memory   = VK_NULL_HANDLE;
        void*           mapped   = nullptr;
    };

    inline auto make_alias_storage(
        const std::shared_ptr<TensorStorage>& slot_storage,
        const shape_t& shape,
        const stride_t& strides,
        DataType dtype,
        std::size_t size_bytes,
        std::size_t offset_bytes,
        DeviceType device
    ) -> std::shared_ptr<TensorStorage> {
        if (device == DeviceType::VULKAN) {
            auto* raw = static_cast<TensorStorage*>(
                ::operator new(sizeof(TensorStorage) + sizeof(VulkanExt)));
            new (raw) TensorStorage{};
            auto* ext = new (reinterpret_cast<VulkanExt*>(raw + 1)) VulkanExt{};
            const auto* slot_ext = reinterpret_cast<const VulkanExt*>(slot_storage.get() + 1);
            ext->buffer = slot_ext->buffer;
            ext->memory = slot_ext->memory;
            ext->mapped = slot_ext->mapped;
            raw->shape = shape;
            raw->strides = strides.empty() ? default_strides(shape) : strides;
            raw->offset_bytes = offset_bytes;
            raw->capacity_bytes = slot_storage ? slot_storage->capacity_bytes : size_bytes;
            raw->dtype = dtype;
            raw->device = DeviceType::VULKAN;
            raw->size_bytes = size_bytes;
            raw->data = slot_ext->mapped ? static_cast<char*>(slot_ext->mapped) + offset_bytes : nullptr;
            raw->external_owner = slot_storage;
            return std::shared_ptr<TensorStorage>(raw, [](TensorStorage* p) {
                if (!p) return;
                p->~TensorStorage();
                reinterpret_cast<VulkanExt*>(p + 1)->~VulkanExt();
                ::operator delete(p);
            });
        }

        auto s = std::make_shared<TensorStorage>();
        s->shape = shape;
        s->strides = strides.empty() ? default_strides(shape) : strides;
        s->offset_bytes = offset_bytes;
        s->capacity_bytes = slot_storage ? slot_storage->capacity_bytes : size_bytes;
        s->dtype = dtype;
        s->device = device;
        s->size_bytes = size_bytes;
        s->data = slot_storage && slot_storage->data ? static_cast<char*>(slot_storage->data) + offset_bytes : nullptr;
        s->external_owner = slot_storage;
        return s;
    }

} // namespace

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

namespace {

    // 活跃与专属张量分析结果
    struct TensorLiveness {
        std::unordered_set<pConstTensorStorage>              active_tensors;
        std::unordered_set<pConstTensorStorage>              dedicated_set;
        std::unordered_map<pConstTensorStorage, std::size_t> birth_map;
        std::unordered_map<pConstTensorStorage, std::size_t> death_map;
    };

    // 别名组规划结果：记录根节点及等价张量成员索引
    struct AliasPlan {
        std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
        std::unordered_set<std::size_t>                           dedicated_roots;
    };

    // 活跃中间张量生命周期活跃区间 [birth, death]
    struct GroupInterval {
        std::size_t              root_idx = 0;
        std::size_t              birth = 0;
        std::size_t              death = 0;
        std::size_t              max_size_bytes = 0;
        std::vector<std::size_t> members;
    };

    // 已规划的中间张量组在单一连续 Arena 中的虚拟区间 [offset, offset + aligned_size)
    struct PlannedGroup {
        GroupInterval interval;
        std::size_t   offset = 0;
        std::size_t   aligned_size = 0;
    };

    // 连续 Arena 内存规划方案：记录所有中间组的虚拟偏移与整块 Arena 的实际所需总容量
    struct ArenaPlan {
        std::vector<PlannedGroup> planned_groups;
        std::size_t               total_size_bytes = 0;
    };

    // 物理对齐约束：满足 Vulkan minStorageBufferOffsetAlignment、AVX-512、CUDA Warp 与缓存行要求
    constexpr std::size_t ARENA_ALIGNMENT = 256;

    inline auto align_up(std::size_t size, std::size_t alignment) -> std::size_t {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    // 物理存储分配结果与指针重映射表
    struct AllocationResult {
        std::vector<std::shared_ptr<TensorStorage>>             staged_tensors;
        std::unordered_map<pConstTensorStorage, pTensorStorage> adopted;
        std::size_t                                             dedicated_allocations_count = 0;

        auto map_storage(pConstTensorStorage storage) const -> pTensorStorage {
            if (auto it = adopted.find(storage); it != adopted.end()) {
                return it->second;
            }
            return const_cast<pTensorStorage>(storage);
        }
    };

    // 判断 Transpose 是否可作为零拷贝跨步视图与下游 MatMul 算子融合
    auto _can_fuse_transpose_as_view(
        pConstTensorStorage out_s,
        const std::vector<int>& perm,
        std::span<const pConstNode> order,
        DeviceType device
    ) -> bool {
        if (perm.size() < 2) return false;
        std::size_t consumer_count = 0;
        for (const auto* other : order) {
            for (std::size_t in_i = 0; in_i < other->inputs().size(); ++in_i) {
                if (other->inputs()[in_i] == out_s) {
                    ++consumer_count;
                    if (other->op().op != Op::MatMul) {
                        return false;
                    }
                    if (device == DeviceType::CUDA) {
                        if (in_i != 1 || perm.size() < 3 ||
                            perm[perm.size() - 2] != static_cast<int>(perm.size() - 1) ||
                            perm[perm.size() - 1] != static_cast<int>(perm.size() - 2)) {
                            return false;
                        }
                    }
                }
            }
        }
        return consumer_count > 0;
    }

    // 收集活跃张量、专属保护集合及各张量的生命周期区间
    template <typename IsInputFn>
    auto _analyze_tensor_liveness(
        std::span<const pConstNode> order,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        bool has_explicit_outputs,
        const std::vector<pConstTensorStorage>& explicit_outputs,
        IsInputFn&& is_input_fn
    ) -> TensorLiveness {
        TensorLiveness result;

        for (auto* node : order) {
            for (auto* in : node->inputs()) result.active_tensors.insert(in);
            for (auto* out : node->outputs()) result.active_tensors.insert(out);
        }
        if (has_explicit_outputs) {
            for (auto* out : explicit_outputs) result.active_tensors.insert(out);
        } else {
            for (const auto& sp : tensors) result.active_tensors.insert(sp.get());
        }

        for (std::size_t i = 0; i < tensors.size(); ++i) {
            auto* sp = tensors[i].get();
            if (sp->data != nullptr || is_input_fn(i)) {
                result.dedicated_set.insert(sp);
            }
        }
        if (has_explicit_outputs) {
            for (auto* out : explicit_outputs) result.dedicated_set.insert(out);
        } else {
            for (auto* sp : result.active_tensors) result.dedicated_set.insert(sp);
        }

        for (std::size_t step = 0; step < order.size(); ++step) {
            for (auto* out : order[step]->outputs()) {
                if (!result.birth_map.contains(out)) {
                    result.birth_map[out] = step;
                }
            }
            for (auto* in : order[step]->inputs()) {
                result.death_map[in] = std::max(result.death_map[in], step);
            }
        }

        return result;
    }

    // 基于并查集建立连续 Reshape 和转置视图的别名组，并向根节点传播专属保护属性
    auto _analyze_alias_groups(
        std::span<const pConstNode> order,
        DeviceType device,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        const std::vector<pConstTensorStorage>& explicit_outputs,
        const std::unordered_set<pConstTensorStorage>& active_tensors,
        std::unordered_set<pConstTensorStorage>& dedicated_set
    ) -> AliasPlan {
        std::unordered_map<pConstTensorStorage, std::size_t> tensor_index_map;
        tensor_index_map.reserve(tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensor_index_map[tensors[i].get()] = i;
        }

        std::vector<std::size_t> alias_parent(tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) alias_parent[i] = i;

        auto find_root = [&](auto& self, std::size_t u) -> std::size_t {
            if (alias_parent[u] == u) return u;
            return alias_parent[u] = self(self, alias_parent[u]);
        };

        for (const auto* node : order) {
            if (node->op().op == Op::Reshape) {
                auto* in_s = node->inputs()[0];
                auto* out_s = node->outputs()[0];
                if (in_s->is_contiguous()) {
                    std::size_t in_idx = tensor_index_map[in_s];
                    std::size_t out_idx = tensor_index_map[out_s];
                    std::size_t r_in = find_root(find_root, in_idx);
                    std::size_t r_out = find_root(find_root, out_idx);
                    if (r_in != r_out) {
                        alias_parent[r_out] = r_in;
                    }
                    out_s->strides = default_strides(out_s->shape);
                }
            } else if (node->op().op == Op::Transpose) {
                if (device == DeviceType::CPU || device == DeviceType::CUDA) {
                    auto* in_s = node->inputs()[0];
                    auto* out_s = node->outputs()[0];
                    bool is_explicit_out = std::find(explicit_outputs.begin(), explicit_outputs.end(), out_s) != explicit_outputs.end();
                    if (in_s->is_contiguous() && !is_explicit_out && !dedicated_set.contains(out_s)) {
                        const auto& tattr = std::get<TransposeAttrs>(node->op().attrs);
                        const auto& perm = tattr.perm;
                        if (_can_fuse_transpose_as_view(out_s, perm, order, device)) {
                            const auto in_strides = in_s->effective_strides();
                            stride_t perm_strides(perm.size());
                            for (std::size_t p = 0; p < perm.size(); ++p) {
                                perm_strides[p] = in_strides[perm[p]];
                            }
                            out_s->strides = std::move(perm_strides);

                            std::size_t in_idx = tensor_index_map[in_s];
                            std::size_t out_idx = tensor_index_map[out_s];
                            std::size_t r_in = find_root(find_root, in_idx);
                            std::size_t r_out = find_root(find_root, out_idx);
                            if (r_in != r_out) {
                                alias_parent[r_out] = r_in;
                            }
                        }
                    }
                }
            }
        }

        AliasPlan plan;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            if (active_tensors.contains(tensors[i].get())) {
                plan.groups[find_root(find_root, i)].push_back(i);
            }
        }

        for (const auto& [root_idx, members] : plan.groups) {
            for (std::size_t m_idx : members) {
                if (dedicated_set.contains(tensors[m_idx].get())) {
                    plan.dedicated_roots.insert(root_idx);
                    break;
                }
            }
        }
        for (std::size_t root_idx : plan.dedicated_roots) {
            for (std::size_t m_idx : plan.groups[root_idx]) {
                dedicated_set.insert(tensors[m_idx].get());
            }
        }

        return plan;
    }

    // 动态存储分配（DSA）虚拟空间规划：
    // 基于 Best-Fit Decreasing by Size（大张量优先），将中间张量生命周期矩形
    // [birth, death] x [offset, offset + aligned_size) 紧凑打包到虚拟单一 Arena 中。
    // 此阶段完全在虚拟偏移空间计算，不申请任何物理设备内存，并确定 Arena 最小总容量。
    auto _plan_arena_offsets(
        const AliasPlan& alias_plan,
        const std::unordered_map<pConstTensorStorage, std::size_t>& birth_map,
        const std::unordered_map<pConstTensorStorage, std::size_t>& death_map,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors
    ) -> ArenaPlan {
        std::vector<GroupInterval> intermediates;
        intermediates.reserve(alias_plan.groups.size());

        for (const auto& [root_idx, members] : alias_plan.groups) {
            if (alias_plan.dedicated_roots.contains(root_idx)) continue;

            std::size_t g_birth = std::numeric_limits<std::size_t>::max();
            std::size_t g_death = 0;
            std::size_t g_max_size = 0;

            for (std::size_t m_idx : members) {
                auto* sp = tensors[m_idx].get();
                if (sp->size_bytes > 0) {
                    auto b_it = birth_map.find(sp);
                    auto d_it = death_map.find(sp);
                    std::size_t b = (b_it != birth_map.end()) ? b_it->second : 0;
                    std::size_t d = (d_it != death_map.end()) ? d_it->second : 0;
                    g_birth = std::min(g_birth, b);
                    g_death = std::max(g_death, std::max(b, d));
                    g_max_size = std::max(g_max_size, sp->size_bytes);
                }
            }

            if (g_max_size > 0 && g_birth <= g_death) {
                intermediates.push_back(GroupInterval{
                    .root_idx = root_idx,
                    .birth = g_birth,
                    .death = g_death,
                    .max_size_bytes = g_max_size,
                    .members = members,
                });
            }
        }

        // 大张量优先排序（Size-First）：先锚定大块内存，防止后续被小张量碎片阻碍
        std::sort(intermediates.begin(), intermediates.end(),
            [](const GroupInterval& a, const GroupInterval& b) {
                if (a.max_size_bytes != b.max_size_bytes) {
                    return a.max_size_bytes > b.max_size_bytes;
                }
                std::size_t dur_a = a.death - a.birth;
                std::size_t dur_b = b.death - b.birth;
                if (dur_a != dur_b) {
                    return dur_a > dur_b;
                }
                if (a.birth != b.birth) {
                    return a.birth < b.birth;
                }
                return a.root_idx < b.root_idx;
            });

        ArenaPlan plan;
        plan.planned_groups.reserve(intermediates.size());

        for (const auto& inter : intermediates) {
            const std::size_t aligned_size = align_up(inter.max_size_bytes, ARENA_ALIGNMENT);

            // 筛选生命周期重叠的时空冲突张量（闭区间 [birth, death] 存在交集）
            std::vector<const PlannedGroup*> overlapping;
            for (const auto& pg : plan.planned_groups) {
                if (std::max(pg.interval.birth, inter.birth) <= std::min(pg.interval.death, inter.death)) {
                    overlapping.push_back(&pg);
                }
            }

            // 候选偏移集：包括 0 以及所有时空冲突组的结束边界
            std::vector<std::size_t> candidates;
            candidates.reserve(overlapping.size() + 1);
            candidates.push_back(0);
            for (const auto* pg : overlapping) {
                candidates.push_back(pg->offset + pg->aligned_size);
            }
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

            // 寻找首个不与任何冲突组重叠的候选偏移
            std::size_t chosen_offset = 0;
            for (std::size_t cand : candidates) {
                bool conflict = false;
                std::size_t cand_end = cand + aligned_size;
                for (const auto* pg : overlapping) {
                    std::size_t pg_end = pg->offset + pg->aligned_size;
                    if (std::max(cand, pg->offset) < std::min(cand_end, pg_end)) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) {
                    chosen_offset = cand;
                    break;
                }
            }

            plan.planned_groups.push_back(PlannedGroup{
                .interval = inter,
                .offset = chosen_offset,
                .aligned_size = aligned_size,
            });
        }

        // 计算整块连续 Arena 所需的最终物理字节数
        for (const auto& pg : plan.planned_groups) {
            plan.total_size_bytes = std::max(plan.total_size_bytes, pg.offset + pg.aligned_size);
        }

        return plan;
    }

    // 校验目标设备后端物理分配器是否就绪
    auto _validate_backend_allocator(
        DeviceType device,
        const ArenaPlan& arena_plan,
        const AliasPlan& alias_plan,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors
    ) -> void {
        bool needs_allocation = arena_plan.total_size_bytes > 0;
        if (!needs_allocation) {
            for (std::size_t root_idx : alias_plan.dedicated_roots) {
                bool has_data = false;
                for (std::size_t m_idx : alias_plan.groups.at(root_idx)) {
                    if (tensors[m_idx]->data != nullptr) {
                        has_data = true;
                        break;
                    }
                }
                if (!has_data) {
                    needs_allocation = true;
                    break;
                }
            }
        }
        if (needs_allocation && internal::get_storage_creator(device) == nullptr) {
            throw std::runtime_error(
                std::string("velomind: no backend allocator registered for device [") +
                device_type_name(device) + "]");
        }
    }

    // 为整块 Arena 分配单次连续物理存储，并为各中间张量创建切片别名
    auto _allocate_arena_storage(
        DeviceType device,
        const ArenaPlan& arena_plan,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        std::size_t fail_after_allocations,
        std::size_t& staged_allocations,
        AllocationResult& result
    ) -> void {
        if (arena_plan.total_size_bytes == 0) return;

        auto arena_buf = TensorStorage::allocate(arena_plan.total_size_bytes, device);
        if (arena_buf == nullptr || arena_buf->device != device ||
            arena_buf->size_bytes < arena_plan.total_size_bytes ||
            arena_buf->data == nullptr) {
            throw std::runtime_error(
                std::string("velomind: backend allocator returned invalid storage for arena ") +
                std::to_string(arena_plan.total_size_bytes) + " bytes on device [" +
                device_type_name(device) + "]");
        }
        if (fail_after_allocations != 0 && ++staged_allocations == fail_after_allocations) {
            throw std::runtime_error("Graph::build: injected allocation-stage failure");
        }

        for (const auto& pg : arena_plan.planned_groups) {
            for (std::size_t m_idx : pg.interval.members) {
                auto* sp = tensors[m_idx].get();
                auto alias = make_alias_storage(
                    arena_buf,
                    sp->shape,
                    sp->strides,
                    sp->dtype,
                    sp->size_bytes,
                    pg.offset,
                    device);
                result.adopted[sp] = alias.get();
                result.staged_tensors[m_idx] = std::move(alias);
            }
        }
    }

    // 为专属活跃组分配物理存储并建立别名映射
    auto _allocate_dedicated_buffers(
        DeviceType device,
        const AliasPlan& alias_plan,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        std::size_t fail_after_allocations,
        std::size_t& staged_allocations,
        AllocationResult& result
    ) -> void {
        for (std::size_t root_idx : alias_plan.dedicated_roots) {
            const auto& members = alias_plan.groups.at(root_idx);
            std::shared_ptr<TensorStorage> base_buf = nullptr;
            for (std::size_t m_idx : members) {
                if (tensors[m_idx]->data != nullptr) {
                    base_buf = tensors[m_idx];
                    break;
                }
            }
            if (base_buf == nullptr) {
                std::size_t max_size = 0;
                for (std::size_t m_idx : members) {
                    max_size = std::max(max_size, tensors[m_idx]->size_bytes);
                }
                base_buf = TensorStorage::allocate(max_size, device);
                if (base_buf == nullptr || base_buf->device != device ||
                    base_buf->size_bytes < max_size ||
                    (max_size > 0 && base_buf->data == nullptr)) {
                    throw std::runtime_error(
                        std::string("velomind: backend allocator returned invalid storage for ") +
                        std::to_string(max_size) + " bytes on device [" +
                        device_type_name(device) + "]");
                }
                base_buf->shape = tensors[root_idx]->shape;
                base_buf->strides = tensors[root_idx]->strides;
                base_buf->dtype = tensors[root_idx]->dtype;
                ++result.dedicated_allocations_count;
                if (fail_after_allocations != 0 && ++staged_allocations == fail_after_allocations) {
                    throw std::runtime_error("Graph::build: injected allocation-stage failure");
                }
            }

            for (std::size_t m_idx : members) {
                auto* sp = tensors[m_idx].get();
                if (m_idx == root_idx) {
                    result.adopted[sp] = base_buf.get();
                    result.staged_tensors[m_idx] = base_buf;
                } else {
                    auto alias = make_alias_storage(base_buf, sp->shape, sp->strides, sp->dtype, sp->size_bytes, 0, device);
                    result.adopted[sp] = alias.get();
                    result.staged_tensors[m_idx] = std::move(alias);
                }
            }
        }
    }

    // 为 0 字节活跃张量准备有效存储对象
    auto _allocate_zero_byte_tensors(
        DeviceType device,
        const std::unordered_set<pConstTensorStorage>& active_tensors,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        AllocationResult& result
    ) -> void {
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto& sp = tensors[i];
            if (sp->data == nullptr && sp->size_bytes == 0 && active_tensors.contains(sp.get()) && !result.adopted.contains(sp.get())) {
                auto new_sp = TensorStorage::allocate(0, device);
                if (new_sp) {
                    new_sp->shape = sp->shape;
                    new_sp->strides = sp->strides;
                    new_sp->dtype = sp->dtype;
                    result.adopted[sp.get()] = new_sp.get();
                    result.staged_tensors[i] = std::move(new_sp);
                }
            }
        }
    }

    // 为 Arena 与专属组物理分配设备存储，并创建视图别名包装
    auto _allocate_planned_storage(
        DeviceType device,
        const ArenaPlan& arena_plan,
        const AliasPlan& alias_plan,
        const std::unordered_set<pConstTensorStorage>& active_tensors,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors
    ) -> AllocationResult {
        _validate_backend_allocator(device, arena_plan, alias_plan, tensors);

        AllocationResult result;
        result.staged_tensors = tensors;
        result.adopted.reserve(tensors.size());

        std::size_t fail_after_allocations = 0;
        if (const char* injected = std::getenv("VELOMIND_FAIL_GRAPH_BUILD_AFTER_ALLOCATIONS")) {
            std::string_view spec(injected);
            const auto parsed = std::from_chars(spec.data(), spec.data() + spec.size(), fail_after_allocations);
            if (parsed.ec != std::errc{} || parsed.ptr != spec.data() + spec.size())
                fail_after_allocations = 0;
        }
        std::size_t staged_allocations = 0;

        _allocate_arena_storage(device, arena_plan, tensors, fail_after_allocations, staged_allocations, result);
        _allocate_dedicated_buffers(device, alias_plan, tensors, fail_after_allocations, staged_allocations, result);
        _allocate_zero_byte_tensors(device, active_tensors, tensors, result);

        return result;
    }

    // 统计内存规划指标（峰值、物理分配次数、复用张量数与节省字节数）
    auto _compute_memory_plan_stats(
        const ArenaPlan& arena_plan,
        const AliasPlan& alias_plan,
        const std::vector<std::shared_ptr<TensorStorage>>& staged_tensors,
        const std::vector<std::shared_ptr<TensorStorage>>& tensors,
        std::size_t dedicated_allocations_count
    ) -> MemoryPlanStats {
        MemoryPlanStats stats;
        stats.peak_bytes = arena_plan.total_size_bytes;
        for (std::size_t root_idx : alias_plan.dedicated_roots) {
            const auto& members = alias_plan.groups.at(root_idx);
            std::size_t g_max = 0;
            for (std::size_t m_idx : members) {
                g_max = std::max(g_max, staged_tensors[m_idx]->size_bytes);
            }
            stats.peak_bytes += g_max;
        }
        stats.allocation_count = (arena_plan.total_size_bytes > 0 ? 1 : 0) + dedicated_allocations_count;

        std::size_t total_intermediates = 0;
        for (const auto& [root_idx, members] : alias_plan.groups) {
            if (!alias_plan.dedicated_roots.contains(root_idx)) {
                total_intermediates += members.size();
            }
        }
        stats.intermediate_tensor_count = total_intermediates;

        // 计算中间张量复用情况：若某个已放置组占用的内存区间与更早消亡的组重叠，则视为复用了物理空间
        for (const auto& pg : arena_plan.planned_groups) {
            bool is_reused_group = false;
            for (const auto& other : arena_plan.planned_groups) {
                if (other.interval.death <= pg.interval.birth) {
                    std::size_t ov_start = std::max(pg.offset, other.offset);
                    std::size_t ov_end = std::min(pg.offset + pg.aligned_size, other.offset + other.aligned_size);
                    if (ov_start < ov_end) {
                        is_reused_group = true;
                        break;
                    }
                }
            }
            if (is_reused_group) {
                stats.reused_tensor_count += pg.interval.members.size();
            } else if (pg.interval.members.size() > 1) {
                stats.reused_tensor_count += (pg.interval.members.size() - 1);
            }
        }

        // 专属组内的别名复用
        for (std::size_t root_idx : alias_plan.dedicated_roots) {
            const auto& members = alias_plan.groups.at(root_idx);
            if (members.size() > 1) {
                stats.reused_tensor_count += (members.size() - 1);
                for (std::size_t i = 1; i < members.size(); ++i) {
                    stats.bytes_saved += tensors[members[i]]->size_bytes;
                }
            }
        }

        // 中间张量节省的字节数：若各组独立分配所需字节总和减去 Arena 实际占用字节
        std::size_t unopt_intermediate_bytes = 0;
        for (const auto& pg : arena_plan.planned_groups) {
            unopt_intermediate_bytes += pg.interval.max_size_bytes;
            if (pg.interval.members.size() > 1) {
                for (std::size_t i = 1; i < pg.interval.members.size(); ++i) {
                    stats.bytes_saved += tensors[pg.interval.members[i]]->size_bytes;
                }
            }
        }
        if (unopt_intermediate_bytes > arena_plan.total_size_bytes) {
            stats.bytes_saved += (unopt_intermediate_bytes - arena_plan.total_size_bytes);
        }

        return stats;
    }

    // 解析算子内核并装配 Executable 拓扑节点
    auto _assemble_executable(
        const Graph* graph,
        const std::shared_ptr<void>& lifetime,
        std::span<const pConstNode> order,
        DeviceType device,
        const AllocationResult& alloc_res,
        const MemoryPlanStats& stats
    ) -> std::unique_ptr<Executable::Impl> {
        auto exec_impl = std::make_unique<Executable::Impl>();
        exec_impl->graph = graph;
        exec_impl->graph_lifetime = lifetime;
        exec_impl->device = device;
        exec_impl->topo.reserve(order.size());
        exec_impl->plan_stats = stats;
    #ifdef VELOMIND_ENABLE_CUDA
        if (device == DeviceType::CUDA) {
            exec_impl->device_context = std::make_shared<backend::cuda::CudaContext>();
        }
    #endif

        for (auto* node : order) {
            Executable::CompiledNode cn;

            for (auto* in : node->inputs()) {
                auto* staged = alloc_res.map_storage(in);
                cn.inputs.push_back(staged);
                cn.input_metadata.push_back(*staged);
            }
            for (auto* out : node->outputs()) {
                auto* staged = alloc_res.map_storage(out);
                cn.outputs.push_back(staged);
                cn.output_metadata.push_back(*staged);
            }

            OpDescriptor desc = node->op();
            cn.attrs = std::shared_ptr<const void>(
                new OpDescriptor(std::move(desc)),
                [](const void* p) { delete reinterpret_cast<const OpDescriptor*>(p); });

            if (node->outputs().empty()) {
                throw std::runtime_error("velomind: op node has no outputs");
            }
            std::vector<DataType> in_dtypes;
            in_dtypes.reserve(node->inputs().size());
            for (auto* in : node->inputs()) in_dtypes.push_back(in->dtype);
            auto* out0 = node->outputs().front();
            cn.kernel = internal::resolve_op_kernel(
                node->op().op,
                in_dtypes,
                out0->dtype,
                device);
            if (cn.kernel == nullptr) {
                throw std::runtime_error(
                    std::string("velomind: no kernel for op=") + op_name(node->op().op));
            }

            exec_impl->topo.push_back(std::move(cn));
        }

        return exec_impl;
    }

} // namespace

auto Graph::build(DeviceType device) -> std::unique_ptr<Executable> {
    if (_built) throw std::logic_error("Graph::build: graph is already built");

    validate(device);
    const auto order = _topological_order();

    auto liveness = _analyze_tensor_liveness(
        order,
        _tensors,
        _has_explicit_outputs,
        _explicit_outputs,
        [this](std::size_t i) { return is_input_tensor(i); }
    );

    auto alias_plan = _analyze_alias_groups(
        order,
        device,
        _tensors,
        _explicit_outputs,
        liveness.active_tensors,
        liveness.dedicated_set
    );

    const auto arena_plan = _plan_arena_offsets(
        alias_plan,
        liveness.birth_map,
        liveness.death_map,
        _tensors
    );

    auto alloc_res = _allocate_planned_storage(
        device,
        arena_plan,
        alias_plan,
        liveness.active_tensors,
        _tensors
    );

    const auto stats = _compute_memory_plan_stats(
        arena_plan,
        alias_plan,
        alloc_res.staged_tensors,
        _tensors,
        alloc_res.dedicated_allocations_count
    );

    auto exec = std::make_unique<Executable>();
    exec->_impl = _assemble_executable(
        this,
        _lifetime,
        order,
        device,
        alloc_res,
        stats
    );

    _tensors.swap(alloc_res.staged_tensors);
    for (auto& node : _nodes) {
        for (auto*& input : node._inputs) input = alloc_res.map_storage(input);
        for (auto*& output : node._outputs) output = alloc_res.map_storage(output);
    }
    for (auto*& out : _explicit_outputs) {
        out = alloc_res.map_storage(out);
    }
    _built = true;
    _bound_device = device;
    _plan_stats = stats;
    return exec;
}

} // namespace velomind
