#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstdint>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

__global__ void embedding_f32_i32_kernel(const float* __restrict__ table,
                                         const std::int32_t* __restrict__ indices,
                                         float* __restrict__ out,
                                         std::size_t num_indices,
                                         std::size_t hidden) {
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = num_indices * hidden;
    if (tid < total) {
        std::size_t token_idx = tid / hidden;
        std::size_t h_idx     = tid % hidden;
        std::int32_t token    = indices[token_idx];
        out[tid] = table[static_cast<std::size_t>(token) * hidden + h_idx];
    }
}

template <typename TTable, typename TIdx, typename TOut>
void embedding_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void*                 ) {
    const auto* table = static_cast<const TTable*>(in[0]->data);
    const auto* idx   = static_cast<const TIdx*>(in[1]->data);
    auto*       y     = static_cast<TOut*>(out[0]->data);

    const std::size_t hidden      = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t num_indices = storage_numel(*in[1]);
    const std::size_t total       = num_indices * hidden;

    if (total == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t blocks = (total + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();

    embedding_f32_i32_kernel<<<blocks, threads, 0, stream>>>(table, idx, y, num_indices, hidden);
    check_cuda_kernel(Op::Embedding, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_embedding_f32(
    DeviceType::CUDA, Op::Embedding,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32, DataType::Int32 }, 2, DataType::Float32 },
    static_cast<Executable::KernelFn>(&embedding_impl<float, std::int32_t, float>));

}

}
