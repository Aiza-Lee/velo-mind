#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"
#include "internal/vulkan_backend.h"

namespace velomind::backend::vulkan {

namespace {

// Vulkan 后端切片内核：在主机可见缓冲区上执行切片复制。
void slice_impl(const TensorStorage* const* in,
                TensorStorage* const*      out,
                const void*                attrs_ptr) {
    backend_vulkan::sync_vulkan_batch_if_pending();

    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<SliceAttrs>(desc.attrs);

    const float* x = static_cast<const float*>(in[0]->data);
    float*       y = static_cast<float*>(out[0]->data);

    const std::size_t out_numel = storage_numel(*out[0]);
    if (out_numel == 0) return;

    const auto& in_shape = in[0]->shape;
    const auto& out_shape = out[0]->shape;
    const std::size_t rank = in_shape.size();

    std::vector<int> b(rank, 0);
    std::vector<int> e(rank, 0);
    std::vector<int> s(rank, 1);

    for (std::size_t d = 0; d < rank; ++d) {
        const int dim = static_cast<int>(in_shape[d]);
        if (d < attr.begins.size()) {
            int begin = attr.begins[d] < 0 ? attr.begins[d] + dim : attr.begins[d];
            int end   = attr.ends[d] < 0 ? attr.ends[d] + dim : attr.ends[d];
            b[d] = std::clamp(begin, 0, dim);
            e[d] = std::clamp(end, 0, dim);
        } else {
            b[d] = 0;
            e[d] = dim;
        }
        if (d < attr.strides.size() && attr.strides[d] > 0) {
            s[d] = attr.strides[d];
        } else {
            s[d] = 1;
        }
    }

    std::size_t K = rank;
    while (K > 0) {
        std::size_t d = K - 1;
        const int dim = static_cast<int>(in_shape[d]);
        if (b[d] == 0 && e[d] == dim && s[d] == 1) {
            K = d;
        } else {
            break;
        }
    }

    std::size_t block_elements = 1;
    for (std::size_t d = K; d < rank; ++d) {
        block_elements *= static_cast<std::size_t>(in_shape[d]);
    }
    const std::size_t block_bytes = block_elements * sizeof(float);

    if (K == 0) {
        std::memcpy(y, x, block_bytes);
        return;
    }

    if (K == 1 && s[0] == 1) {
        const std::size_t in_offset = static_cast<std::size_t>(b[0]) * block_elements;
        const std::size_t num_bytes = static_cast<std::size_t>(e[0] - b[0]) * block_bytes;
        std::memcpy(y, x + in_offset, num_bytes);
        return;
    }

    std::vector<std::size_t> in_strides(rank, 1);
    for (std::size_t d = rank - 1; d-- > 0;) {
        in_strides[d] = in_strides[d + 1] * static_cast<std::size_t>(in_shape[d + 1]);
    }

    std::size_t outer_blocks = 1;
    for (std::size_t d = 0; d < K; ++d) {
        outer_blocks *= static_cast<std::size_t>(out_shape[d]);
    }

    for (std::size_t m = 0; m < outer_blocks; ++m) {
        std::size_t rem = m;
        std::size_t in_idx = 0;
        for (std::size_t d = K; d-- > 0;) {
            std::size_t o_d = rem % static_cast<std::size_t>(out_shape[d]);
            rem /= static_cast<std::size_t>(out_shape[d]);
            in_idx += (static_cast<std::size_t>(b[d]) + o_d * static_cast<std::size_t>(s[d])) * in_strides[d];
        }
        std::memcpy(y + m * block_elements, x + in_idx, block_bytes);
    }
}

static ::velomind::internal::KernelRegistrar
    _velomind_kr_slice_f32_vk(
        DeviceType::VULKAN,
        Op::Slice,
        ::velomind::internal::KernelDtypeKey{
            { DataType::Float32 },
            1,
            DataType::Float32
        },
        static_cast<Executable::KernelFn>(&slice_impl));

} // namespace

VELOMIND_REGISTER_SIGNATURE(Op::Slice, ::velomind::detail::signature_slice)

} // namespace velomind::backend::vulkan
