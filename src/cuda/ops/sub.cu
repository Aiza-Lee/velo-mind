#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"

namespace velomind::backend::cuda {

template <typename T>
__global__ void sub_kernel(const T* a, const T* b, T* c, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

template <typename T>
void sub_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T* a = static_cast<const T*>(in[0]->data);
    const T* b = static_cast<const T*>(in[1]->data);
    T*       c = static_cast<T*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    if (n == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();
    sub_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);
    check_cuda_kernel(Op::Sub, *out[0]);
}

namespace {
VELOMIND_REGISTER_BINARY_1T(DeviceType::CUDA, Op::Sub, DataType::Float32, sub_impl);
}

}
