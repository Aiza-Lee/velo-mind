#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "backend_error.h"
#include "internal/registry/kernel.h"

namespace velomind::backend::cuda {

namespace {

// CUDA GQA 头分组重复内核：每个输出线程根据自身全局索引逆向求解对应的 KV 头并执行复制。
template <typename T>
__global__ void repeat_kv_kernel(const T* __restrict__ x, T* __restrict__ y,
                                 std::size_t outer,
                                 std::size_t kv_heads,
                                 std::size_t repeats,
                                 std::size_t inner) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t q_heads = kv_heads * repeats;
    std::size_t total = outer * q_heads * inner;
    if (idx >= total) return;

    std::size_t in_idx = idx % inner;
    std::size_t rem = idx / inner;
    std::size_t q_head_idx = rem % q_heads;
    std::size_t out_idx = rem / q_heads;

    std::size_t kv_head_idx = q_head_idx / repeats;
    std::size_t src = (out_idx * kv_heads + kv_head_idx) * inner + in_idx;
    y[idx] = x[src];
}

template <typename T>
void repeat_kv_cuda_impl(const TensorStorage* const* in,
                         TensorStorage* const*      out,
                         const void*                attrs_ptr) {
    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<RepeatKVAttrs>(desc.attrs);

    const T* x = static_cast<const T*>(in[0]->data);
    T*       y = static_cast<T*>(out[0]->data);

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

    std::size_t total = outer * (kv_heads * repeats) * inner;
    if (total == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t blocks = (total + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();
    repeat_kv_kernel<<<blocks, threads, 0, stream>>>(x, y, outer, kv_heads, repeats, inner);

    check_cuda_kernel(Op::RepeatKV, *out[0]);
}

VELOMIND_REGISTER_UNARY_1T_FLOATS(DeviceType::CUDA, Op::RepeatKV, repeat_kv_cuda_impl);


}

}
