#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cuda_runtime.h>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

__global__ void rsqrt_f32_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = rsqrtf(x[i]);
    }
}

template <typename T>
void rsqrt_impl(const TensorStorage* const* in,
                TensorStorage* const*      out,
                const void*                 ) {
    const auto* x = static_cast<const T*>(in[0]->data);
    auto*       y = static_cast<T*>(out[0]->data);

    std::size_t n = storage_numel(*out[0]);
    if (n == 0) return;

    constexpr std::size_t threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();

    rsqrt_f32_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
    check_cuda_kernel(Op::Rsqrt, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_rsqrt_f32(
    DeviceType::CUDA, Op::Rsqrt,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32 }, 1, DataType::Float32 },
    static_cast<Executable::KernelFn>(&rsqrt_impl<float>));

}

}
