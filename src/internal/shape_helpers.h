#pragma once

#include <algorithm>
#include <span>

#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "internal/registry/shape.h"
#include "internal/softmax.h"

namespace velomind::detail {

inline auto signature_softmax(std::span<const pConstTensorStorage> in,
                             const OpAttrs& attrs) -> ::velomind::internal::OpSignature {
    if (in.size() != 1 || !in[0]) throw std::invalid_argument("Softmax: expected one input");
    const auto* softmax = std::get_if<SoftmaxAttrs>(&attrs);
    if (!softmax) throw std::invalid_argument("Softmax: invalid attributes");
    const bool valid_dtype = in[0]->dtype == DataType::Float32 ||
                             in[0]->dtype == DataType::Float16 ||
                             in[0]->dtype == DataType::BFloat16;
    if (!valid_dtype) throw std::invalid_argument("Softmax: expected Float32, Float16, or BFloat16");
    softmax_layout(in[0]->shape, *softmax);
    return {.dtype = in[0]->dtype, .shape = in[0]->shape};
}

inline auto signature_elementwise(std::span<const pConstTensorStorage> in,
                                  const OpAttrs& ) -> ::velomind::internal::OpSignature {
    return ::velomind::internal::OpSignature{
        .dtype = in.empty() ? DataType::Float32 : in[0]->dtype,
        .shape = in.empty() ? shape_t{} : in[0]->shape,
    };
}

inline auto signature_matmul(std::span<const pConstTensorStorage> in,
                             const OpAttrs& ) -> ::velomind::internal::OpSignature {
    const auto& a_shape = in[0]->shape;
    const auto& b_shape = in[1]->shape;
    const std::size_t a_rank = a_shape.size();
    const std::size_t b_rank = b_shape.size();
    if (a_rank < 2 || b_rank < 2) {
        throw std::invalid_argument(
            "velomind::MatMul signature: inputs must be rank >= 2");
    }

    // 在前方补 1 后统一比较批次维；当前不支持批次广播。
    ::velomind::shape_t a_pad = a_shape;
    ::velomind::shape_t b_pad = b_shape;
    const std::size_t rank = std::max(a_rank, b_rank);
    while (a_pad.size() < rank) a_pad.insert(a_pad.begin(), 1);
    while (b_pad.size() < rank) b_pad.insert(b_pad.begin(), 1);

    ::velomind::shape_t out_shape;
    out_shape.reserve(rank);

    for (std::size_t i = 0; i + 2 < rank; ++i) {
        if (a_pad[i] != b_pad[i]) {
            throw std::invalid_argument(
                "velomind::MatMul signature: batch dim mismatch");
        }
        out_shape.push_back(a_pad[i]);
    }

    if (a_pad[rank - 1] != b_pad[rank - 2]) {
        throw std::invalid_argument(
            "velomind::MatMul signature: K dimension mismatch");
    }
    out_shape.push_back(a_pad[rank - 2]);
    out_shape.push_back(b_pad[rank - 1]);

    DataType out_dtype = in[0]->dtype;
    if (in[0]->dtype == DataType::Float32 || in[1]->dtype == DataType::Float32) {
        if (in[0]->dtype == DataType::Float16 || in[1]->dtype == DataType::Float16 ||
            in[0]->dtype == DataType::BFloat16 || in[1]->dtype == DataType::BFloat16) {
            out_dtype = DataType::Float32;
        }
    }

    return ::velomind::internal::OpSignature{
        .dtype = out_dtype,
        .shape = std::move(out_shape),
    };
}

inline auto signature_rmsnorm(std::span<const pConstTensorStorage> in,
                              const OpAttrs& ) -> ::velomind::internal::OpSignature {
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = in[0]->shape,
    };
}

inline auto signature_embedding(std::span<const pConstTensorStorage> in,
                                const OpAttrs& ) -> ::velomind::internal::OpSignature {
    ::velomind::shape_t out_shape;
    out_shape.reserve(in[1]->shape.size() + 1);
    for (auto d : in[1]->shape) out_shape.push_back(d);
    out_shape.push_back(in[0]->shape.back());
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

inline auto signature_rotary_embedding(std::span<const pConstTensorStorage> in,
                                       const OpAttrs& ) -> ::velomind::internal::OpSignature {
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = in[0]->shape,
    };
}

inline auto signature_reshape(std::span<const pConstTensorStorage> in,
                              const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    const auto& r = std::get<ReshapeAttrs>(attrs);
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = r.shape,
    };
}

inline auto signature_transpose(std::span<const pConstTensorStorage> in,
                                const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    const auto& t = std::get<TransposeAttrs>(attrs);
    ::velomind::shape_t out_shape = in[0]->shape;
    if (t.perm.empty()) {
        // 空排列沿用 NumPy 约定：反转全部轴。

        for (std::size_t i = 0, j = out_shape.size(); i < j; ++i) {
            out_shape[i] = in[0]->shape[j - 1 - i];
        }
    } else {
        for (std::size_t i = 0; i < t.perm.size(); ++i) {
            out_shape[i] = in[0]->shape[t.perm[i]];
        }
    }
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

inline auto signature_concat(std::span<const pConstTensorStorage> in,
                             const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    const auto& c  = std::get<ConcatAttrs>(attrs);
    const int axis = c.axis < 0 ? c.axis + static_cast<int>(in[0]->shape.size())
                                : c.axis;
    ::velomind::shape_t out_shape = in[0]->shape;
    out_shape[axis] += in[1]->shape[axis];
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

inline auto signature_repeat_kv(std::span<const pConstTensorStorage> in,
                                const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    const auto& a = std::get<RepeatKVAttrs>(attrs);
    const int rank = static_cast<int>(in[0]->shape.size());
    const int axis = a.axis < 0 ? a.axis + rank : a.axis;
    ::velomind::shape_t out_shape = in[0]->shape;
    out_shape[axis] *= static_cast<dim_t>(a.repeats);
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

inline auto signature_slice(std::span<const pConstTensorStorage> in,
                            const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    if (in.size() != 1 || !in[0]) {
        throw std::invalid_argument("Slice: expected single input tensor");
    }
    const auto& s = std::get<SliceAttrs>(attrs);
    const auto& in_shape = in[0]->shape;
    const std::size_t rank = in_shape.size();
    ::velomind::shape_t out_shape;
    out_shape.reserve(rank);

    for (std::size_t i = 0; i < rank; ++i) {
        const int dim = static_cast<int>(in_shape[i]);
        if (i < s.begins.size()) {
            const int stride = (i < s.strides.size() && s.strides[i] > 0) ? s.strides[i] : 1;
            int b = s.begins[i] < 0 ? s.begins[i] + dim : s.begins[i];
            int e = s.ends[i] < 0 ? s.ends[i] + dim : s.ends[i];
            b = std::clamp(b, 0, dim);
            e = std::clamp(e, 0, dim);
            const int out_d = (e > b) ? (e - b + stride - 1) / stride : 0;
            out_shape.push_back(static_cast<dim_t>(out_d));
        } else {
            out_shape.push_back(in_shape[i]);
        }
    }
    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

// 融合注意力接口契约：Q, K, V 输入张量要求秩相同且至少为 3（[..., H, S, D]）；
// 支持标准多头及 GQA（H_q % H_k == 0，H_k == H_v）；D_q == D_k；因果模式下要求 S_k >= S_q。
inline auto signature_fused_attention(std::span<const pConstTensorStorage> in,
                                      const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    if (in.size() != 3 || !in[0] || !in[1] || !in[2]) {
        throw std::invalid_argument("FusedAttention: expected 3 inputs (Q, K, V)");
    }
    const auto* fa = std::get_if<FusedAttentionAttrs>(&attrs);
    if (!fa) {
        throw std::invalid_argument("FusedAttention: invalid attributes");
    }
    const auto& q_shape = in[0]->shape;
    const auto& k_shape = in[1]->shape;
    const auto& v_shape = in[2]->shape;
    if (q_shape.size() < 3 || q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size()) {
        throw std::invalid_argument("FusedAttention: inputs must have matching rank >= 3");
    }
    const std::size_t rank = q_shape.size();
    for (std::size_t i = 0; i + 3 < rank; ++i) {
        if (q_shape[i] != k_shape[i] || q_shape[i] != v_shape[i]) {
            throw std::invalid_argument("FusedAttention: batch dimension mismatch");
        }
    }
    const dim_t H_q = q_shape[rank - 3];
    const dim_t H_k = k_shape[rank - 3];
    const dim_t H_v = v_shape[rank - 3];
    if (H_k != H_v || H_k <= 0 || H_q % H_k != 0) {
        throw std::invalid_argument("FusedAttention: head count mismatch or invalid GQA ratio");
    }
    const dim_t S_q = q_shape[rank - 2];
    const dim_t S_k = k_shape[rank - 2];
    const dim_t S_v = v_shape[rank - 2];
    if (S_k != S_v) {
        throw std::invalid_argument("FusedAttention: K and V sequence lengths must match");
    }
    if (fa->is_causal && S_k < S_q) {
        throw std::invalid_argument("FusedAttention: causal attention requires S_k >= S_q");
    }
    const dim_t D_q = q_shape[rank - 1];
    const dim_t D_k = k_shape[rank - 1];
    const dim_t D_v = v_shape[rank - 1];
    if (D_q != D_k) {
        throw std::invalid_argument("FusedAttention: Q and K feature dimensions must match");
    }

    ::velomind::shape_t out_shape = q_shape;
    out_shape[rank - 1] = D_v;

    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

// 量化矩阵乘法接口契约：A [..., M, K], B_q [K, N] 或 Int4 打包 [(K+1)/2, N], scales [N] 或 [num_blocks, N]
inline auto signature_quantized_matmul(std::span<const pConstTensorStorage> in,
                                       const OpAttrs&                       attrs)
    -> ::velomind::internal::OpSignature
{
    if (in.size() != 3 || !in[0] || !in[1] || !in[2]) {
        throw std::invalid_argument("QuantizedMatMul: expected 3 inputs (A, B_q, scales)");
    }
    const auto* qa = std::get_if<QuantizedMatMulAttrs>(&attrs);
    if (!qa) {
        throw std::invalid_argument("QuantizedMatMul: invalid attributes");
    }
    const auto& a_shape = in[0]->shape;
    const auto& b_shape = in[1]->shape;
    const auto& s_shape = in[2]->shape;

    if (a_shape.size() < 2) {
        throw std::invalid_argument("QuantizedMatMul: activation rank must be >= 2");
    }
    if (b_shape.size() != 2) {
        throw std::invalid_argument("QuantizedMatMul: quantized weight must be 2D matrix");
    }

    const dim_t k_a = a_shape.back();
    dim_t k_b = b_shape[0];
    const dim_t n   = b_shape[1];

    if (qa->quant_type == QuantType::Int4) {
        const dim_t expected_packed_rows = (k_a + 1) / 2;
        if (b_shape[0] != expected_packed_rows) {
            throw std::invalid_argument("QuantizedMatMul (Int4): packed rows mismatch with activation inner dimension");
        }
    } else {
        if (k_a != k_b) {
            throw std::invalid_argument("QuantizedMatMul: inner dimension mismatch between activation and weight");
        }
    }

    if (qa->block_size > 0) {
        const dim_t num_blocks = (k_a + qa->block_size - 1) / qa->block_size;
        if (s_shape.size() == 2) {
            if (s_shape[0] != num_blocks || s_shape[1] != n) {
                throw std::invalid_argument("QuantizedMatMul: block-wise scales shape mismatch");
            }
        } else if (s_shape.size() == 1) {
            if (s_shape[0] != num_blocks * n) {
                throw std::invalid_argument("QuantizedMatMul: flat block-wise scales size mismatch");
            }
        } else {
            throw std::invalid_argument("QuantizedMatMul: invalid scales rank");
        }
    } else {
        if (s_shape.size() == 1) {
            if (s_shape[0] != n) {
                throw std::invalid_argument("QuantizedMatMul: per-channel scales size mismatch with output columns");
            }
        } else if (s_shape.size() == 2) {
            if (s_shape[0] != 1 || s_shape[1] != n) {
                throw std::invalid_argument("QuantizedMatMul: 2D per-channel scales shape mismatch");
            }
        } else {
            throw std::invalid_argument("QuantizedMatMul: invalid scales rank");
        }
    }

    ::velomind::shape_t out_shape = a_shape;
    out_shape.back() = n;

    return ::velomind::internal::OpSignature{
        .dtype = in[0]->dtype,
        .shape = std::move(out_shape),
    };
}

} // namespace velomind::detail
