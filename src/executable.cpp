#include "velomind/executable.h"

#include "velomind/graph.h"
#include "velomind/tensor_storage.h"
#include "internal/registry/memory_transfer.h"

#include <cstdint>

#ifdef VELOMIND_ENABLE_CUDA
#include "cuda/cuda_context.h"
#endif
#ifdef VELOMIND_ENABLE_VULKAN
#include "internal/vulkan_backend.h"
#endif

namespace velomind {

Executable::~Executable() {
#ifdef VELOMIND_ENABLE_CUDA
    if (_impl && _impl->device == DeviceType::CUDA && _impl->device_context) {
        try {
            static_cast<backend::cuda::CudaContext*>(_impl->device_context.get())->synchronize();
        } catch (...) {}
    }
#endif
#ifdef VELOMIND_ENABLE_VULKAN
    if (_impl && _impl->device == DeviceType::VULKAN) {
        try {
            backend_vulkan::sync_vulkan_batch_if_pending();
        } catch (...) {}
    }
#endif
}
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;

auto Executable::execute() -> void {
    if (!_impl) throw std::logic_error("Executable::execute: not built");
    if (_impl->graph_lifetime.expired())
        throw std::logic_error("Executable::execute: owning graph has been destroyed");
    const auto validate = [](const TensorStorage& current, const TensorStorage& compiled) {
        if (current.shape != compiled.shape || current.dtype != compiled.dtype ||
            current.device != compiled.device || current.size_bytes != compiled.size_bytes)
            throw std::invalid_argument("Executable::execute: tensor metadata changed after build");
        if (storage_nbytes(current) > 0 && !current.data)
            throw std::invalid_argument("Executable::execute: tensor buffer is null");
    };
    // 先验证全部节点，避免后段非法元数据使前段输出被部分覆盖。
    for (const auto& cn : _impl->topo) {
        for (std::size_t i = 0; i < cn.inputs.size(); ++i) validate(*cn.inputs[i], cn.input_metadata[i]);
        for (std::size_t i = 0; i < cn.outputs.size(); ++i) validate(*cn.outputs[i], cn.output_metadata[i]);
    }

#ifdef VELOMIND_ENABLE_CUDA
    std::unique_ptr<backend::cuda::ScopedCudaContext> scoped_cuda;
    if (_impl->device == DeviceType::CUDA && _impl->device_context) {
        auto* cuda_ctx = static_cast<backend::cuda::CudaContext*>(_impl->device_context.get());
        scoped_cuda = std::make_unique<backend::cuda::ScopedCudaContext>(cuda_ctx);
    }
#endif
#ifdef VELOMIND_ENABLE_VULKAN
    std::unique_ptr<backend_vulkan::ScopedVulkanBatch> scoped_vulkan;
    if (_impl->device == DeviceType::VULKAN) {
        scoped_vulkan = std::make_unique<backend_vulkan::ScopedVulkanBatch>();
    }
#endif

    for (auto& cn : _impl->topo) {
        const auto& desc = *static_cast<const OpDescriptor*>(cn.attrs.get());
        if (desc.op == Op::Embedding) {
            const auto& indices = *cn.inputs[1];
            const auto count = storage_numel(indices);
            std::vector<std::int32_t> host_indices(count);
            if (count != 0) {
#ifdef VELOMIND_ENABLE_CUDA
                if (_impl->device == DeviceType::CUDA && _impl->device_context) {
                    static_cast<backend::cuda::CudaContext*>(_impl->device_context.get())->synchronize();
                }
#endif
#ifdef VELOMIND_ENABLE_VULKAN
                if (_impl->device == DeviceType::VULKAN) {
                    backend_vulkan::sync_vulkan_batch_if_pending();
                }
#endif
                const auto transfer = internal::get_memory_transfer(indices.device);
                if (!transfer.copy_d2h) throw std::runtime_error("Embedding: no index readback transfer");
                transfer.copy_d2h(host_indices.data(), indices.data, count * sizeof(std::int32_t));
            }
            // 索引可能由前序节点生成，因此必须在每次 gather 前验证实际值。
            for (auto index : host_indices)
                if (index < 0 || index >= cn.inputs[0]->shape[0])
                    throw std::invalid_argument("Embedding: index out of vocabulary range");
        }
        cn.kernel(cn.inputs.data(), cn.outputs.data(), cn.attrs.get());
    }

#ifdef VELOMIND_ENABLE_CUDA
    // 执行边界同步：确保非阻塞流上所有异步算子完成后再返回主机
    if (_impl->device == DeviceType::CUDA && _impl->device_context) {
        static_cast<backend::cuda::CudaContext*>(_impl->device_context.get())->synchronize();
    }
#endif
}

auto Executable::bind_input(Tensor t, ExternalBuffer external_buffer) -> void {
    if (!_impl) throw std::logic_error("Executable::bind_input: not built");
    if (_impl->graph_lifetime.expired())
        throw std::logic_error("Executable::bind_input: owning graph has been destroyed");
    if (t.graph() != _impl->graph || !t.storage())
        throw std::invalid_argument("Executable::bind_input: invalid or foreign tensor");
    auto* g = const_cast<Graph*>(static_cast<const Graph*>(_impl->graph));
    if (!g->is_input_tensor(t.index()))
        throw std::invalid_argument("Executable::bind_input: tensor is not a graph input");
    auto* s = g->tensor_storage_at(t.index());
    if (external_buffer.device != _impl->device || s->device != _impl->device)
        throw std::invalid_argument("Executable::bind_input: device mismatch");
    if (_impl->device == DeviceType::VULKAN)
        throw std::invalid_argument("Executable::bind_input: Vulkan external buffers require VkBuffer binding");
    if (external_buffer.capacity_bytes < s->size_bytes)
        throw std::invalid_argument("Executable::bind_input: insufficient buffer capacity");
    if (s->size_bytes > 0 && (!external_buffer.data || !external_buffer.owner))
        throw std::invalid_argument("Executable::bind_input: nonempty buffer requires data and owner");
    if (external_buffer.owner && external_buffer.owner.get() != external_buffer.data)
        throw std::invalid_argument("Executable::bind_input: owner must alias buffer data");
    if (external_buffer.data &&
        reinterpret_cast<std::uintptr_t>(external_buffer.data) % data_type_size(s->dtype) != 0)
        throw std::invalid_argument("Executable::bind_input: unaligned buffer");
    s->external_owner = std::move(external_buffer.owner);
    s->data = external_buffer.data;
}

auto Executable::output(Tensor t) -> pTensorStorage {
    if (!_impl || _impl->graph_lifetime.expired() || t.graph() == nullptr) return nullptr;
    if (t.graph() != _impl->graph) return nullptr;
    auto* g = const_cast<Graph*>(static_cast<const Graph*>(_impl->graph));
    return g->tensor_storage_at(t.index());
}

auto Executable::output(Tensor t) const -> pConstTensorStorage {
    if (!_impl || _impl->graph_lifetime.expired() || t.graph() == nullptr) return nullptr;
    if (t.graph() != _impl->graph) return nullptr;
    return _impl->graph->tensor_storage_at(t.index());
}

auto Executable::num_nodes() const -> std::size_t {
    return _impl ? _impl->topo.size() : 0;
}

auto Executable::memory_plan_stats() const noexcept -> const MemoryPlanStats& {
    static const MemoryPlanStats empty;
    return _impl ? _impl->plan_stats : empty;
}

} // namespace velomind
