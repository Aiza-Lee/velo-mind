#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "velomind/dtype.h"
#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/quant.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind::backend::cuda {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

__device__ inline auto sign_extend_4bit_dev(std::uint8_t nibble) noexcept -> std::int8_t {
    if (nibble & 0x08) {
        return static_cast<std::int8_t>(nibble | 0xF0);
    }
    return static_cast<std::int8_t>(nibble & 0x0F);
}

// M=1 自回归解码单 token GEMV Warp 协作核函数
template <typename TAct, typename TScale, typename TOut>
__global__ void quantized_gemv_int8_warp_kernel(
    const TAct* __restrict__        a,
    const std::int8_t* __restrict__ b_q,
    const TScale* __restrict__      scale,
    TOut* __restrict__              c,
    std::size_t                     k,
    std::size_t                     n,
    std::size_t                     total_outputs)
{
    const std::size_t warp_id = (static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK) +
                                (threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= total_outputs) return;

    const std::size_t col = warp_id % n;
    const std::size_t batch_idx = warp_id / n;

    const TAct* a_vec = a + batch_idx * k;

    float acc = 0.0f;
    for (std::size_t p = lane_id; p < k; p += WARP_SIZE) {
        const float a_val = static_cast<float>(a_vec[p]);
        const float w_val = static_cast<float>(b_q[p * n + col]);
        acc += a_val * w_val;
    }

    // Warp 树状快速规约
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        const float s = static_cast<float>(scale[col]);
        c[batch_idx * n + col] = static_cast<TOut>(acc * s);
    }
}

// M=1 自回归解码单 token INT4 GEMV Warp 协作核函数
template <typename TAct, typename TScale, typename TOut>
__global__ void quantized_gemv_int4_warp_kernel(
    const TAct* __restrict__         a,
    const std::uint8_t* __restrict__ b_packed,
    const TScale* __restrict__       scale,
    TOut* __restrict__               c,
    std::size_t                      k,
    std::size_t                      n,
    std::size_t                      block_size,
    std::size_t                      total_outputs)
{
    const std::size_t warp_id = (static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK) +
                                (threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= total_outputs) return;

    const std::size_t col = warp_id % n;
    const std::size_t batch_idx = warp_id / n;

    const TAct* a_vec = a + batch_idx * k;
    const std::size_t bs = (block_size > 0) ? block_size : 32;
    const std::size_t num_blocks = (k + bs - 1) / bs;

    float total_acc = 0.0f;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t p_start = b * bs;
        const std::size_t p_end   = (p_start + bs < k) ? (p_start + bs) : k;

        float block_acc = 0.0f;
        for (std::size_t p = p_start + lane_id; p < p_end; p += WARP_SIZE) {
            const std::uint8_t byte_val = b_packed[(p / 2) * n + col];
            const std::uint8_t nibble = (p % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            const float w_val = static_cast<float>(sign_extend_4bit_dev(nibble));
            const float a_val = static_cast<float>(a_vec[p]);
            block_acc += a_val * w_val;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }

        if (lane_id == 0) {
            const float s = static_cast<float>(scale[b * n + col]);
            total_acc += block_acc * s;
        }
    }

    if (lane_id == 0) {
        c[batch_idx * n + col] = static_cast<TOut>(total_acc);
    }
}

// M>1 通用 2D GEMM 二维网格内核
template <typename TAct, typename TScale, typename TOut>
__global__ void quantized_gemm_2d_kernel(
    const TAct* __restrict__        a,
    const std::int8_t* __restrict__ b_q,
    const TScale* __restrict__      scale,
    TOut* __restrict__              c,
    std::size_t                     m,
    std::size_t                     k,
    std::size_t                     n,
    QuantType                       quant_type,
    int                             block_size)
{
    const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    const TAct* a_row = a + row * k;

    if (quant_type == QuantType::Int4) {
        const auto* b_packed = reinterpret_cast<const std::uint8_t*>(b_q);
        const std::size_t bs = (block_size > 0) ? static_cast<std::size_t>(block_size) : 32;
        const std::size_t num_blocks = (k + bs - 1) / bs;

        float total_acc = 0.0f;
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t p_start = b * bs;
            const std::size_t p_end   = (p_start + bs < k) ? (p_start + bs) : k;
            const float s = static_cast<float>(scale[b * n + col]);

            float block_acc = 0.0f;
            for (std::size_t p = p_start; p < p_end; ++p) {
                const std::uint8_t byte_val = b_packed[(p / 2) * n + col];
                const std::uint8_t nibble = (p % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
                const float w_val = static_cast<float>(sign_extend_4bit_dev(nibble));
                const float a_val = static_cast<float>(a_row[p]);
                block_acc += a_val * w_val;
            }
            total_acc += block_acc * s;
        }
        c[row * n + col] = static_cast<TOut>(total_acc);
        return;
    }

    if (block_size > 0) {
        const auto bs = static_cast<std::size_t>(block_size);
        const std::size_t num_blocks = (k + bs - 1) / bs;

        float total_acc = 0.0f;
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t p_start = b * bs;
            const std::size_t p_end   = (p_start + bs < k) ? (p_start + bs) : k;
            const float s = static_cast<float>(scale[b * n + col]);

            float block_acc = 0.0f;
            for (std::size_t p = p_start; p < p_end; ++p) {
                const float a_val = static_cast<float>(a_row[p]);
                const float w_val = static_cast<float>(b_q[p * n + col]);
                block_acc += a_val * w_val;
            }
            total_acc += block_acc * s;
        }
        c[row * n + col] = static_cast<TOut>(total_acc);
    } else {
        float acc = 0.0f;
        for (std::size_t p = 0; p < k; ++p) {
            const float a_val = static_cast<float>(a_row[p]);
            const float w_val = static_cast<float>(b_q[p * n + col]);
            acc += a_val * w_val;
        }
        const float s = static_cast<float>(scale[col]);
        c[row * n + col] = static_cast<TOut>(acc * s);
    }
}

template <typename TAct, typename TScale, typename TOut>
void quantized_matmul_cuda_impl(
    const pConstTensorStorage* in,
    const pTensorStorage*      out,
    const void*                attrs)
{
    const auto* act_stor    = in[0];
    const auto* weight_stor = in[1];
    const auto* scale_stor  = in[2];
    auto*       out_stor    = out[0];

    if (!act_stor || !weight_stor || !scale_stor || !out_stor) {
        throw std::runtime_error("quantized_matmul cuda: null storage");
    }

    const auto& desc = *static_cast<const OpDescriptor*>(attrs);
    const auto& attr = std::get<QuantizedMatMulAttrs>(desc.attrs);

    const auto* a = static_cast<const TAct*>(act_stor->data);
    const auto* b_q = static_cast<const std::int8_t*>(weight_stor->data);
    const auto* s = static_cast<const TScale*>(scale_stor->data);
    auto*       c = static_cast<TOut*>(out_stor->data);

    const auto& a_shape = act_stor->shape;
    const auto& b_shape = weight_stor->shape;
    const auto& c_shape = out_stor->shape;

    const std::size_t a_rank = a_shape.size();
    const std::size_t c_rank = c_shape.size();

    const std::size_t m = static_cast<std::size_t>(a_shape[a_rank - 2]);
    const std::size_t k = static_cast<std::size_t>(a_shape[a_rank - 1]);
    const std::size_t n = static_cast<std::size_t>(b_shape[1]);

    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < c_rank; ++i) {
        batch *= static_cast<std::size_t>(c_shape[i]);
    }

    auto stream = get_cuda_context().stream_handle();

    if (m == 1) {
        const std::size_t total_outputs = batch * n;
        const std::size_t blocks = (total_outputs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        if (attr.quant_type == QuantType::Int4) {
            const auto* b_packed = reinterpret_cast<const std::uint8_t*>(b_q);
            quantized_gemv_int4_warp_kernel<TAct, TScale, TOut><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                a, b_packed, s, c, k, n,
                static_cast<std::size_t>(attr.block_size > 0 ? attr.block_size : 32),
                total_outputs);
        } else {
            quantized_gemv_int8_warp_kernel<TAct, TScale, TOut><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                a, b_q, s, c, k, n, total_outputs);
        }
    } else {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (m * batch + block.y - 1) / block.y);

        quantized_gemm_2d_kernel<TAct, TScale, TOut><<<grid, block, 0, stream>>>(
            a, b_q, s, c, m * batch, k, n, attr.quant_type, attr.block_size);
    }

    check_cuda_kernel(Op::QuantizedMatMul, *out[0]);
}

VELOMIND_REGISTER_QUANTIZED_MATMUL(DeviceType::CUDA, DataType::Float32,  DataType::Float32,  DataType::Float32,  quantized_matmul_cuda_impl);
VELOMIND_REGISTER_QUANTIZED_MATMUL(DeviceType::CUDA, DataType::Float16,  DataType::Float16,  DataType::Float16,  quantized_matmul_cuda_impl);
VELOMIND_REGISTER_QUANTIZED_MATMUL(DeviceType::CUDA, DataType::Float16,  DataType::Float32,  DataType::Float16,  quantized_matmul_cuda_impl);
VELOMIND_REGISTER_QUANTIZED_MATMUL(DeviceType::CUDA, DataType::BFloat16, DataType::BFloat16, DataType::BFloat16, quantized_matmul_cuda_impl);
VELOMIND_REGISTER_QUANTIZED_MATMUL(DeviceType::CUDA, DataType::BFloat16, DataType::Float32,  DataType::BFloat16, quantized_matmul_cuda_impl);

} // namespace

} // namespace velomind::backend::cuda
