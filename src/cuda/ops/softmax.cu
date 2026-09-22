#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/dtype.h"

#include <cmath>
#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/softmax.h"

namespace velomind::backend::cuda {

template <typename T>
__global__ void softmax_kernel(const T* x, T* y, std::size_t rows, std::size_t cols, std::size_t inner) {

    std::size_t row = blockIdx.x;
    if (row >= rows) return;
    const T* x_row = x + (row / inner) * cols * inner + row % inner;
    T*       y_row = y + (row / inner) * cols * inner + row % inner;

    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        float m = static_cast<float>(x_row[0]);
        for (std::size_t j = 1; j < cols; ++j) {
            float v = static_cast<float>(x_row[j * inner]);
            if (v > m) m = v;
        }
        shared_max = m;
    }
    __syncthreads();
    float row_max = shared_max;

    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            s += expf(static_cast<float>(x_row[j * inner]) - row_max);
        }
        shared_sum = s;
    }
    __syncthreads();
    float row_sum = shared_sum;

    for (std::size_t j = 0; j < cols; ++j) {
        float val = expf(static_cast<float>(x_row[j * inner]) - row_max) / row_sum;
        y_row[j * inner] = static_cast<T>(val);
    }
}

template <typename T>
void softmax_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* attrs) {
    const T* x = static_cast<const T*>(in[0]->data);
    T*       y = static_cast<T*>(out[0]->data);

    const auto layout = detail::softmax_layout(*in[0], attrs);
    const auto cols = layout.cols;
    const auto inner = layout.inner;
    const auto rows = layout.outer * inner;
    if (rows == 0 || cols == 0) return;

    auto stream = get_cuda_context().stream_handle();
    softmax_kernel<<<rows, 1, 0, stream>>>(x, y, rows, cols, inner);
    check_cuda_kernel(Op::Softmax, *out[0]);
}

namespace {
    static ::velomind::internal::KernelRegistrar _velomind_kr_softmax_f32(
        DeviceType::CUDA, Op::Softmax,
        ::velomind::internal::KernelDtypeKey{
            { DataType::Float32 }, 1, DataType::Float32 },
        static_cast<Executable::KernelFn>(&softmax_impl<float>));

    static ::velomind::internal::KernelRegistrar _velomind_kr_softmax_f16(
        DeviceType::CUDA, Op::Softmax,
        ::velomind::internal::KernelDtypeKey{
            { DataType::Float16 }, 1, DataType::Float16 },
        static_cast<Executable::KernelFn>(&softmax_impl<float16_t>));

    static ::velomind::internal::KernelRegistrar _velomind_kr_softmax_bf16(
        DeviceType::CUDA, Op::Softmax,
        ::velomind::internal::KernelDtypeKey{
            { DataType::BFloat16 }, 1, DataType::BFloat16 },
        static_cast<Executable::KernelFn>(&softmax_impl<bfloat16_t>));
}

}
