#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/dtype.h"

#include <cmath>
#include <cstddef>
#include <variant>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

template <typename T1, typename T2, typename T3>
__global__ void rmsnorm_kernel(const T1* __restrict__ x,
                               const T2* __restrict__ weight,
                               T3* __restrict__ y,
                               std::size_t outer,
                               std::size_t reduce_dim,
                               float eps) {
    std::size_t row = blockIdx.x;
    if (row >= outer) return;

    const T1* x_row = x + row * reduce_dim;
    T3*       y_row = y + row * reduce_dim;

    float thread_sumsq = 0.0f;
    for (std::size_t i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        float v = static_cast<float>(x_row[i]);
        thread_sumsq += v * v;
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = thread_sumsq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / static_cast<float>(reduce_dim) + eps);
    __syncthreads();

    for (std::size_t i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]) * inv_rms * static_cast<float>(weight[i]);
        y_row[i] = static_cast<T3>(val);
    }
}

template <typename T1, typename T2, typename T3>
void rmsnorm_impl(const TensorStorage* const* in,
                  TensorStorage* const*      out,
                  const void*                 attrs_ptr) {
    float eps = 1e-5f;
    if (attrs_ptr != nullptr) {
        const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
        if (std::holds_alternative<RMSNormAttrs>(desc.attrs)) {
            eps = static_cast<float>(std::get<RMSNormAttrs>(desc.attrs).epsilon);
        }
    }

    const auto* x      = static_cast<const T1*>(in[0]->data);
    const auto* weight = static_cast<const T2*>(in[1]->data);
    auto*       y      = static_cast<T3*>(out[0]->data);

    const auto& shape = in[0]->shape;
    if (shape.empty()) return;

    const std::size_t reduce_dim = static_cast<std::size_t>(shape.back());
    const std::size_t outer      = storage_numel(*in[0]) / reduce_dim;

    if (outer == 0 || reduce_dim == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t shared_bytes = threads * sizeof(float);
    auto stream = get_cuda_context().stream_handle();

    rmsnorm_kernel<T1, T2, T3><<<outer, threads, shared_bytes, stream>>>(
        x, weight, y, outer, reduce_dim, eps);
    check_cuda_kernel(Op::RMSNorm, *out[0]);
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::RMSNorm, DataType::Float32,  DataType::Float32,  DataType::Float32,  rmsnorm_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::RMSNorm, DataType::Float16,  DataType::Float16,  DataType::Float16,  rmsnorm_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::RMSNorm, DataType::BFloat16, DataType::BFloat16, DataType::BFloat16, rmsnorm_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::RMSNorm, DataType::Float32,  DataType::Float16,  DataType::Float32,  rmsnorm_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::RMSNorm, DataType::Float32,  DataType::BFloat16, DataType::Float32,  rmsnorm_impl);

}

}
