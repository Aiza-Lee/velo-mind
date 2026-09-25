#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"

namespace velomind::backend::cuda {

template <typename T1, typename T2, typename T3>
__global__ void add_kernel(const T1* a, const T2* b, T3* c, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = static_cast<T3>(a[i] + b[i]);
}

template <typename T1, typename T2, typename T3>
void add_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T1* a = static_cast<const T1*>(in[0]->data);
    const T2* b = static_cast<const T2*>(in[1]->data);
    T3*       c = static_cast<T3*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    if (n == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);

    check_cuda_kernel(Op::Add, *out[0]);
}

VELOMIND_REGISTER_BINARY_SAME(
    DeviceType::CUDA,
    Op::Add,
    add_impl
);

}
