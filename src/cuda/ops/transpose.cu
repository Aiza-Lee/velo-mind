#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <variant>
#include <vector>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

struct TransposeParams {
    std::size_t in_strides[8];
    std::size_t out_strides[8];
    int         perm[8];
    std::size_t rank;
    std::size_t total;
};

template <typename T>
__global__ void transpose_kernel(const T* __restrict__ x,
                                 T* __restrict__ y,
                                 TransposeParams p) {
    std::size_t flat = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < p.total) {
        std::size_t rem = flat;
        std::size_t in_flat = 0;
        for (std::size_t ax = 0; ax < p.rank; ++ax) {
            std::size_t coord = rem / p.out_strides[ax];
            rem %= p.out_strides[ax];
            in_flat += coord * p.in_strides[p.perm[ax]];
        }
        y[flat] = x[in_flat];
    }
}

template <typename T>
void transpose_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void*                 attrs_ptr) {
    const auto* x = static_cast<const T*>(in[0]->data);
    auto*       y = static_cast<T*>(out[0]->data);

    const auto& in_shape  = in[0]->shape;
    const auto& out_shape = out[0]->shape;
    const std::size_t rank = in_shape.size();

    if (x == y || rank == 0 || rank > 8) return;

    std::vector<int> perm;
    if (attrs_ptr != nullptr) {
        const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
        if (std::holds_alternative<TransposeAttrs>(desc.attrs)) {
            perm = std::get<TransposeAttrs>(desc.attrs).perm;
        }
    }
    if (perm.empty()) {
        perm.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            perm[i] = static_cast<int>(rank - 1 - i);
        }
    }

    TransposeParams p{};
    p.rank  = rank;
    p.total = storage_numel(*out[0]);

    if (p.total == 0) return;

    for (std::size_t i = 0; i < 8; ++i) {
        p.in_strides[i]  = 1;
        p.out_strides[i] = 1;
        p.perm[i]        = static_cast<int>(i);
    }

    const auto raw_in_strides = in[0]->effective_strides();
    for (std::size_t i = 0; i < rank; ++i) {
        p.in_strides[i] = static_cast<std::size_t>(raw_in_strides[i]);
    }

    p.out_strides[rank - 1] = 1;
    for (std::size_t i = rank - 1; i > 0; --i) {
        p.out_strides[i - 1] = p.out_strides[i] * static_cast<std::size_t>(out_shape[i]);
    }

    for (std::size_t i = 0; i < rank; ++i) {
        p.perm[i] = perm[i];
    }

    constexpr std::size_t threads = 256;
    std::size_t blocks = (p.total + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();

    transpose_kernel<<<blocks, threads, 0, stream>>>(x, y, p);
    check_cuda_kernel(Op::Transpose, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_transpose_f32(
    DeviceType::CUDA, Op::Transpose,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32 }, 1, DataType::Float32 },
    static_cast<Executable::KernelFn>(&transpose_impl<float>));

static ::velomind::internal::KernelRegistrar _velomind_kr_transpose_f16(
    DeviceType::CUDA, Op::Transpose,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float16 }, 1, DataType::Float16 },
    static_cast<Executable::KernelFn>(&transpose_impl<float16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_transpose_bf16(
    DeviceType::CUDA, Op::Transpose,
    ::velomind::internal::KernelDtypeKey{
        { DataType::BFloat16 }, 1, DataType::BFloat16 },
    static_cast<Executable::KernelFn>(&transpose_impl<bfloat16_t>));

}

}
