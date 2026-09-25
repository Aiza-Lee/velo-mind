#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"

namespace velomind::backend::cuda {

template <typename T>
__global__ void sigmoid_kernel(const T* x, T* y, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = T(1) / (T(1) + exp(-x[i]));
}

template <typename T>
void sigmoid_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T* x = static_cast<const T*>(in[0]->data);
    T*       y = static_cast<T*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    if (n == 0) return;
    constexpr std::size_t threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    auto stream = get_cuda_context().stream_handle();
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
    check_cuda_kernel(Op::Sigmoid, *out[0]);
}

namespace {
VELOMIND_REGISTER_UNARY_1T(DeviceType::CUDA, Op::Sigmoid, DataType::Float32, sigmoid_impl);
}

}
