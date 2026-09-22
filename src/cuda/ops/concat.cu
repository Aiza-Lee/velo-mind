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

template <typename T>
__global__ void concat_kernel(const T* a, const T* b, T* out,
                              std::size_t outer,
                              std::size_t a_axis_len,
                              std::size_t b_axis_len,
                              std::size_t inner) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t out_axis_len = a_axis_len + b_axis_len;
    std::size_t total = outer * out_axis_len * inner;
    if (idx >= total) return;

    std::size_t in_idx = idx % inner;
    std::size_t rem = idx / inner;
    std::size_t ax_idx = rem % out_axis_len;
    std::size_t out_idx = rem / out_axis_len;

    if (ax_idx < a_axis_len) {
        std::size_t src = (out_idx * a_axis_len + ax_idx) * inner + in_idx;
        out[idx] = a[src];
    } else {
        std::size_t src = (out_idx * b_axis_len + (ax_idx - a_axis_len)) * inner + in_idx;
        out[idx] = b[src];
    }
}

template <typename T>
void concat_cuda_impl(const TensorStorage* const* in,
                      TensorStorage* const*      out,
                      const void*                 attrs_ptr) {
    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& cattr = std::get<ConcatAttrs>(desc.attrs);

    const T* a = static_cast<const T*>(in[0]->data);
    const T* b = static_cast<const T*>(in[1]->data);
    T*       y = static_cast<T*>(out[0]->data);

    const auto& a_shape = in[0]->shape;
    const std::size_t rank = a_shape.size();

    int axis = cattr.axis < 0 ? cattr.axis + static_cast<int>(rank) : cattr.axis;

    std::size_t outer = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
        outer *= static_cast<std::size_t>(a_shape[i]);
    }
    const std::size_t a_axis_len = static_cast<std::size_t>(a_shape[axis]);
    const std::size_t b_axis_len = static_cast<std::size_t>(in[1]->shape[axis]);
    const std::size_t out_axis_len = a_axis_len + b_axis_len;
    std::size_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < rank; ++i) {
        inner *= static_cast<std::size_t>(a_shape[i]);
    }

    std::size_t total = outer * out_axis_len * inner;
    if (total == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t blocks = (total + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();
    concat_kernel<<<blocks, threads, 0, stream>>>(a, b, y, outer, a_axis_len, b_axis_len, inner);

    check_cuda_kernel(Op::Concat, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_concat_f32(
    DeviceType::CUDA, Op::Concat,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32, DataType::Float32 }, 2, DataType::Float32 },
    static_cast<Executable::KernelFn>(&concat_cuda_impl<float>));

static ::velomind::internal::KernelRegistrar _velomind_kr_concat_f16(
    DeviceType::CUDA, Op::Concat,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float16, DataType::Float16 }, 2, DataType::Float16 },
    static_cast<Executable::KernelFn>(&concat_cuda_impl<float16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_concat_bf16(
    DeviceType::CUDA, Op::Concat,
    ::velomind::internal::KernelDtypeKey{
        { DataType::BFloat16, DataType::BFloat16 }, 2, DataType::BFloat16 },
    static_cast<Executable::KernelFn>(&concat_cuda_impl<bfloat16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_concat_i32(
    DeviceType::CUDA, Op::Concat,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Int32, DataType::Int32 }, 2, DataType::Int32 },
    static_cast<Executable::KernelFn>(&concat_cuda_impl<int32_t>));

} // namespace

} // namespace velomind::backend::cuda
