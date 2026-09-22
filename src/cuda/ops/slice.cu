#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

#include "backend_error.h"
#include "internal/registry/kernel.h"

namespace velomind::backend::cuda {

namespace {

struct CudaSliceDims {
    int rank = 0;
    int b[8] = {0};
    int s[8] = {1};
    int in_strides[8] = {0};
    int out_shape[8] = {0};
};

template <typename T>
__global__ void slice_kernel(const T* __restrict__ x, T* __restrict__ y,
                             CudaSliceDims dims, std::size_t total) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    std::size_t rem = idx;
    std::size_t in_idx = 0;
    #pragma unroll
    for (int d = 7; d >= 0; --d) {
        if (d >= dims.rank) continue;
        int o_d = rem % dims.out_shape[d];
        rem /= dims.out_shape[d];
        in_idx += static_cast<std::size_t>(dims.b[d] + o_d * dims.s[d]) * dims.in_strides[d];
    }
    y[idx] = x[in_idx];
}

template <typename T>
void slice_cuda_impl(const TensorStorage* const* in,
                     TensorStorage* const*      out,
                     const void*                attrs_ptr) {
    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<SliceAttrs>(desc.attrs);

    const T* x = static_cast<const T*>(in[0]->data);
    T*       y = static_cast<T*>(out[0]->data);

    const std::size_t out_numel = storage_numel(*out[0]);
    if (out_numel == 0) return;

    const auto& in_shape = in[0]->shape;
    const auto& out_shape = out[0]->shape;
    const std::size_t rank = in_shape.size();
    if (rank > 8) {
        throw std::invalid_argument("Slice: CUDA kernel supports up to rank 8");
    }

    CudaSliceDims dims;
    dims.rank = static_cast<int>(rank);

    for (std::size_t d = 0; d < rank; ++d) {
        const int dim = static_cast<int>(in_shape[d]);
        if (d < attr.begins.size()) {
            int begin = attr.begins[d] < 0 ? attr.begins[d] + dim : attr.begins[d];
            int end   = attr.ends[d] < 0 ? attr.ends[d] + dim : attr.ends[d];
            dims.b[d] = std::clamp(begin, 0, dim);
        } else {
            dims.b[d] = 0;
        }
        if (d < attr.strides.size() && attr.strides[d] > 0) {
            dims.s[d] = attr.strides[d];
        } else {
            dims.s[d] = 1;
        }
        dims.out_shape[d] = static_cast<int>(out_shape[d]);
    }

    dims.in_strides[rank - 1] = 1;
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
        dims.in_strides[d] = dims.in_strides[d + 1] * static_cast<int>(in_shape[d + 1]);
    }

    // 检查是否为首轴连续整切片（如末行视图），若是直接使用异步 D2D 内存拷贝
    bool is_axis0_contiguous = (dims.s[0] == 1);
    for (std::size_t d = 1; d < rank; ++d) {
        if (dims.b[d] != 0 || dims.out_shape[d] != static_cast<int>(in_shape[d]) || dims.s[d] != 1) {
            is_axis0_contiguous = false;
            break;
        }
    }

    auto stream = get_cuda_context().stream_handle();
    if (is_axis0_contiguous) {
        std::size_t in_offset = static_cast<std::size_t>(dims.b[0]) * dims.in_strides[0];
        check_cuda(
            cudaMemcpyAsync(y, x + in_offset, out_numel * sizeof(T), cudaMemcpyDeviceToDevice, stream),
            Op::Slice, *out[0], "memcpy_d2d");

        if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
            const auto key = std::string("CUDA:Slice:sync");
            if (key == injected) check_cuda(cudaSuccess, Op::Slice, *out[0], "sync");
        }
        if (is_cuda_sync_debug_enabled()) {
            check_cuda(cudaStreamSynchronize(stream), Op::Slice, *out[0], "sync");
        }
        return;
    }

    constexpr std::size_t threads = 256;
    std::size_t blocks = (out_numel + threads - 1) / threads;
    slice_kernel<<<blocks, threads, 0, stream>>>(x, y, dims, out_numel);

    check_cuda_kernel(Op::Slice, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_slice_f32(
    DeviceType::CUDA, Op::Slice,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32 }, 1, DataType::Float32 },
    static_cast<Executable::KernelFn>(&slice_cuda_impl<float>));

static ::velomind::internal::KernelRegistrar _velomind_kr_slice_f16(
    DeviceType::CUDA, Op::Slice,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float16 }, 1, DataType::Float16 },
    static_cast<Executable::KernelFn>(&slice_cuda_impl<float16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_slice_bf16(
    DeviceType::CUDA, Op::Slice,
    ::velomind::internal::KernelDtypeKey{
        { DataType::BFloat16 }, 1, DataType::BFloat16 },
    static_cast<Executable::KernelFn>(&slice_cuda_impl<bfloat16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_slice_i32(
    DeviceType::CUDA, Op::Slice,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Int32 }, 1, DataType::Int32 },
    static_cast<Executable::KernelFn>(&slice_cuda_impl<std::int32_t>));

} // namespace

} // namespace velomind::backend::cuda
