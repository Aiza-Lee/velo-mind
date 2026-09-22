#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

__global__ void rotary_embedding_f32_kernel(const float* __restrict__ x,
                                            const float* __restrict__ cos_cache,
                                            const float* __restrict__ sin_cache,
                                            float* __restrict__ y,
                                            std::size_t total_pairs,
                                            std::size_t head_dim,
                                            std::size_t half_dim,
                                            std::size_t cos_rows) {
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < total_pairs) {
        std::size_t row = tid / half_dim;
        std::size_t j   = tid % half_dim;
        std::size_t pos = (cos_rows > 1) ? (row % cos_rows) : 0;

        const float* x_row   = x + row * head_dim;
        float*       y_row   = y + row * head_dim;
        const float* cos_row = cos_cache + pos * half_dim;
        const float* sin_row = sin_cache + pos * half_dim;

        float a = x_row[j];
        float b = x_row[j + half_dim];
        float c = cos_row[j];
        float s = sin_row[j];

        y_row[j]            = a * c - b * s;
        y_row[j + half_dim] = b * c + a * s;
    }
}

template <typename TIn, typename TCos, typename TSin>
void rotary_embedding_impl(const TensorStorage* const* in,
                           TensorStorage* const*      out,
                           const void*                 ) {
    const auto* x   = static_cast<const TIn*>(in[0]->data);
    const auto* cos = static_cast<const TCos*>(in[1]->data);
    const auto* sin = static_cast<const TSin*>(in[2]->data);
    auto*       y   = static_cast<TIn*>(out[0]->data);

    const std::size_t head_dim    = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t half_dim    = head_dim / 2;
    const std::size_t outer       = storage_numel(*in[0]) / head_dim;
    const std::size_t cos_rows    = (in[1]->shape.size() >= 2) ? static_cast<std::size_t>(in[1]->shape[0]) : 1;
    const std::size_t total_pairs = outer * half_dim;

    if (total_pairs == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t blocks = (total_pairs + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();

    rotary_embedding_f32_kernel<<<blocks, threads, 0, stream>>>(
        x, cos, sin, y, total_pairs, head_dim, half_dim, cos_rows);
    check_cuda_kernel(Op::RotaryEmbedding, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_rotary_embedding(
    DeviceType::CUDA, Op::RotaryEmbedding,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32, DataType::Float32, DataType::Float32 }, 3, DataType::Float32 },
    static_cast<Executable::KernelFn>(&rotary_embedding_impl<float, float, float>));

}

}
