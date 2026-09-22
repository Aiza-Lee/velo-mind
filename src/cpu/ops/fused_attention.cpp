#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/dtype.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

// 基于在线 Softmax 流式计算单头/多头注意力输出，消除中间 [H, S_q, S_k] 分数矩阵与权重的全量显存分配。
template <typename T>
void fused_attention_impl(const TensorStorage* const* in,
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

    const std::size_t q_head_stride = S_q * D;
    const std::size_t k_head_stride = S_k * D;
    const std::size_t v_head_stride = S_k * D_v;
    const std::size_t o_head_stride = S_q * D_v;

    const std::size_t q_batch_stride = H_q * q_head_stride;
    const std::size_t k_batch_stride = H_k * k_head_stride;
    const std::size_t v_batch_stride = H_k * v_head_stride;
    const std::size_t o_batch_stride = H_q * o_head_stride;

    std::vector<float> acc(D_v);

    for (std::size_t b = 0; b < outer; ++b) {
        const T* q_batch = q_ptr + b * q_batch_stride;
        const T* k_batch = k_ptr + b * k_batch_stride;
        const T* v_batch = v_ptr + b * v_batch_stride;
        T*       o_batch = o_ptr + b * o_batch_stride;

        for (std::size_t h = 0; h < H_q; ++h) {
            const std::size_t h_kv = h / G;
            const T* q_head = q_batch + h * q_head_stride;
            const T* k_head = k_batch + h_kv * k_head_stride;
            const T* v_head = v_batch + h_kv * v_head_stride;
            T*       o_head = o_batch + h * o_head_stride;

            for (std::size_t i = 0; i < S_q; ++i) {
                const T* q_row = q_head + i * D;
                T*       o_row = o_head + i * D_v;

                const dim_t max_k = is_causal
                    ? (static_cast<dim_t>(S_k - S_q) + static_cast<dim_t>(i) + 1)
                    : static_cast<dim_t>(S_k);
                const std::size_t valid_k = static_cast<std::size_t>(
                    std::clamp<dim_t>(max_k, 0, static_cast<dim_t>(S_k)));

                if (valid_k == 0) {
                    for (std::size_t d = 0; d < D_v; ++d) {
                        o_row[d] = static_cast<T>(0);
                    }
                    continue;
                }

                float m = -std::numeric_limits<float>::infinity();
                float l = 0.0f;
                std::fill(acc.begin(), acc.end(), 0.0f);

                for (std::size_t j = 0; j < valid_k; ++j) {
                    const T* k_row = k_head + j * D;
                    const T* v_row = v_head + j * D_v;

                    float dot = 0.0f;
                    for (std::size_t d = 0; d < D; ++d) {
                        dot += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
                    }
                    float score = dot * scale;

                    if (score > m) {
                        float alpha = std::exp(m - score);
                        l = l * alpha + 1.0f;
                        for (std::size_t d = 0; d < D_v; ++d) {
                            acc[d] = acc[d] * alpha + static_cast<float>(v_row[d]);
                        }
                        m = score;
                    } else {
                        float p = std::exp(score - m);
                        l += p;
                        for (std::size_t d = 0; d < D_v; ++d) {
                            acc[d] += p * static_cast<float>(v_row[d]);
                        }
                    }
                }

                const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
                for (std::size_t d = 0; d < D_v; ++d) {
                    o_row[d] = static_cast<T>(acc[d] * inv_l);
                }
            }
        }
    }
}

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_f32(
    DeviceType::CPU, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float32, DataType::Float32, DataType::Float32 }, 3, DataType::Float32 },
    &fused_attention_impl<float>);

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_f16(
    DeviceType::CPU, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::Float16, DataType::Float16, DataType::Float16 }, 3, DataType::Float16 },
    &fused_attention_impl<float16_t>);

static ::velomind::internal::KernelRegistrar _velomind_kr_fused_attention_bf16(
    DeviceType::CPU, Op::FusedAttention,
    ::velomind::internal::KernelDtypeKey{
        { DataType::BFloat16, DataType::BFloat16, DataType::BFloat16 }, 3, DataType::BFloat16 },
    &fused_attention_impl<bfloat16_t>);

} // namespace

VELOMIND_REGISTER_SIGNATURE(Op::FusedAttention, ::velomind::detail::signature_fused_attention)

} // namespace velomind::backend::cpu
