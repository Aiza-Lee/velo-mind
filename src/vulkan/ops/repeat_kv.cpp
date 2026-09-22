#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"
#include "internal/vulkan_backend.h"

namespace velomind::backend::vulkan {

namespace {

// Vulkan 后端头复制内核：主机可见缓冲区上的连续块内存复制。
void repeat_kv_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void*                attrs_ptr) {
    backend_vulkan::sync_vulkan_batch_if_pending();

    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<RepeatKVAttrs>(desc.attrs);

    const float* x = static_cast<const float*>(in[0]->data);
    float*       y = static_cast<float*>(out[0]->data);

    if (storage_numel(*out[0]) == 0) return;

    const auto& in_shape = in[0]->shape;
    const std::size_t rank = in_shape.size();
    const int axis = attr.axis < 0 ? attr.axis + static_cast<int>(rank) : attr.axis;
    const std::size_t repeats = static_cast<std::size_t>(attr.repeats);
    const std::size_t kv_heads = static_cast<std::size_t>(in_shape[axis]);

    std::size_t outer = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
        outer *= static_cast<std::size_t>(in_shape[i]);
    }
    std::size_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < rank; ++i) {
        inner *= static_cast<std::size_t>(in_shape[i]);
    }

    const std::size_t q_heads = kv_heads * repeats;
    const std::size_t inner_bytes = inner * sizeof(float);

    for (std::size_t o = 0; o < outer; ++o) {
        const float* in_outer  = x + o * kv_heads * inner;
        float*       out_outer = y + o * q_heads * inner;
        for (std::size_t kv = 0; kv < kv_heads; ++kv) {
            const float* src = in_outer + kv * inner;
            for (std::size_t r = 0; r < repeats; ++r) {
                float* dst = out_outer + (kv * repeats + r) * inner;
                std::memcpy(dst, src, inner_bytes);
            }
        }
    }
}

static ::velomind::internal::KernelRegistrar
    _velomind_kr_repeat_kv_f32_vk(
        DeviceType::VULKAN,
        Op::RepeatKV,
        ::velomind::internal::KernelDtypeKey{
            { DataType::Float32 },
            1,
            DataType::Float32
        },
        static_cast<Executable::KernelFn>(&repeat_kv_impl));

}

}
