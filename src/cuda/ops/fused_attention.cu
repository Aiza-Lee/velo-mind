#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/dtype.h"

#include <cmath>
#include <cstddef>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int MAX_ITEMS_PER_LANE = 8; // 支持最高 32 * 8 = 256 维度的 head_dim

// 每个 warp 协作处理一行 Query (batch, head, query_seq_i)
template <typename T>
__global__ void fused_attention_warp_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T*       __restrict__ o,
    std::size_t outer,
    std::size_t H_q,
    std::size_t H_k,
    std::size_t G,
    std::size_t S_q,
    std::size_t S_k,
    std::size_t D,
    std::size_t D_v,
    float       scale,
    bool        is_causal,
    std::size_t total_rows)
{
    const std::size_t warp_id = (static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK) +
                                (threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= total_rows) return;

    // 分解 warp_id -> (b, h, i)
    const std::size_t i   = warp_id % S_q;
    const std::size_t rem = warp_id / S_q;
    const std::size_t h   = rem % H_q;
    const std::size_t b   = rem / H_q;

    const std::size_t h_kv = h / G;

    const std::size_t q_head_stride = S_q * D;
    const std::size_t k_head_stride = S_k * D;
    const std::size_t v_head_stride = S_k * D_v;
    const std::size_t o_head_stride = S_q * D_v;

    const std::size_t q_batch_stride = H_q * q_head_stride;
    const std::size_t k_batch_stride = H_k * k_head_stride;
    const std::size_t v_batch_stride = H_k * v_head_stride;
    const std::size_t o_batch_stride = H_q * o_head_stride;

    const T* q_row  = q + b * q_batch_stride + h * q_head_stride + i * D;
    const T* k_base = k + b * k_batch_stride + h_kv * k_head_stride;
    const T* v_base = v + b * v_batch_stride + h_kv * v_head_stride;
    T*       o_row  = o + b * o_batch_stride + h * o_head_stride + i * D_v;

    // 当前 query 的因果可见 key 范围
    const long long max_k = is_causal
        ? (static_cast<long long>(S_k - S_q) + static_cast<long long>(i) + 1)
        : static_cast<long long>(S_k);
    const std::size_t valid_k = (max_k <= 0) ? 0 : ((static_cast<std::size_t>(max_k) < S_k) ? static_cast<std::size_t>(max_k) : S_k);

    const int num_q_items = static_cast<int>((D + WARP_SIZE - 1) / WARP_SIZE);
    const int num_v_items = static_cast<int>((D_v + WARP_SIZE - 1) / WARP_SIZE);

    if (valid_k == 0) {
        for (int idx = 0; idx < num_v_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
            const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
            if (d < D_v) {
                o_row[d] = static_cast<T>(0.0f);
            }
        }
        return;
    }

    // 每个 lane 缓存其负责维度的 Q 分量
    float q_val[MAX_ITEMS_PER_LANE];
    for (int idx = 0; idx < num_q_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
        const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
        q_val[idx] = (d < D) ? static_cast<float>(q_row[d]) : 0.0f;
    }

    float acc[MAX_ITEMS_PER_LANE];
    for (int idx = 0; idx < MAX_ITEMS_PER_LANE; ++idx) {
        acc[idx] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (std::size_t j = 0; j < valid_k; ++j) {
        const T* k_row = k_base + j * D;
        const T* v_row = v_base + j * D_v;

        // 计算当前线程负责维度的局部内积分量
        float partial = 0.0f;
        for (int idx = 0; idx < num_q_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
            const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
            if (d < D) {
                partial += q_val[idx] * static_cast<float>(k_row[d]);
            }
        }

        // Warp 级归约累加得到完整的内积标量
        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            partial += __shfl_xor_sync(0xffffffff, partial, mask);
        }
        const float score = partial * scale;

        // 在线 Softmax 更新与数值归一化
        if (score > m) {
            const float alpha = expf(m - score);
            l = l * alpha + 1.0f;
            for (int idx = 0; idx < num_v_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
                const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
                if (d < D_v) {
                    acc[idx] = acc[idx] * alpha + static_cast<float>(v_row[d]);
                }
            }
            m = score;
        } else {
            const float p = expf(score - m);
            l += p;
            for (int idx = 0; idx < num_v_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
                const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
                if (d < D_v) {
                    acc[idx] += p * static_cast<float>(v_row[d]);
                }
            }
        }
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int idx = 0; idx < num_v_items && idx < MAX_ITEMS_PER_LANE; ++idx) {
        const std::size_t d = lane_id + static_cast<std::size_t>(idx) * WARP_SIZE;
        if (d < D_v) {
            o_row[d] = static_cast<T>(acc[idx] * inv_l);
        }
    }
}

template <typename T>
void fused_attention_cuda_impl(const TensorStorage* const* in,
                               TensorStorage* const*      out,
                               const void*                attrs_ptr) {
    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<FusedAttentionAttrs>(desc.attrs);

    const auto* q_ptr = static_cast<const T*>(in[0]->data);
    const auto* k_ptr = static_cast<const T*>(in[1]->data);
    const auto* v_ptr = static_cast<const T*>(in[2]->data);
    auto*       o_ptr = static_cast<T*>(out[0]->data);

    if (storage_numel(*out[0]) == 0) return;

    const auto& q_shape = in[0]->shape;
    const auto& k_shape = in[1]->shape;
    const auto& v_shape = in[2]->shape;
    const std::size_t rank = q_shape.size();

    std::size_t outer = 1;
    for (std::size_t i = 0; i + 3 < rank; ++i) {
        outer *= static_cast<std::size_t>(q_shape[i]);
    }

    const std::size_t H_q = static_cast<std::size_t>(q_shape[rank - 3]);
    const std::size_t H_k = static_cast<std::size_t>(k_shape[rank - 3]);
    const std::size_t G   = H_q / H_k;
    const std::size_t S_q = static_cast<std::size_t>(q_shape[rank - 2]);
    const std::size_t S_k = static_cast<std::size_t>(k_shape[rank - 2]);
    const std::size_t D   = static_cast<std::size_t>(q_shape[rank - 1]);
    const std::size_t D_v = static_cast<std::size_t>(v_shape[rank - 1]);

    const float scale = (attr.scale > 0.0f)
        ? attr.scale
        : (1.0f / std::sqrt(static_cast<float>(D)));
    const bool is_causal = attr.is_causal;

    const std::size_t total_rows = outer * H_q * S_q;
    if (total_rows == 0) return;

    const std::size_t blocks = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const std::size_t threads = WARPS_PER_BLOCK * WARP_SIZE;

    auto stream = get_cuda_context().stream_handle();

    fused_attention_warp_kernel<T><<<blocks, threads, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr,
        outer, H_q, H_k, G, S_q, S_k, D, D_v,
        scale, is_causal, total_rows);

    check_cuda_kernel(Op::FusedAttention, *out[0]);
}

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_cuda_f32(
    DeviceType::CUDA, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32, DataType::Float32, DataType::Float32 }, 3, DataType::Float32 },
    static_cast<Executable::KernelFn>(&fused_attention_cuda_impl<float>));

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_cuda_f16(
    DeviceType::CUDA, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float16, DataType::Float16, DataType::Float16 }, 3, DataType::Float16 },
    static_cast<Executable::KernelFn>(&fused_attention_cuda_impl<float16_t>));

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_cuda_bf16(
    DeviceType::CUDA, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::BFloat16, DataType::BFloat16, DataType::BFloat16 }, 3, DataType::BFloat16 },
    static_cast<Executable::KernelFn>(&fused_attention_cuda_impl<bfloat16_t>));

} // namespace

} // namespace velomind::backend::cuda
