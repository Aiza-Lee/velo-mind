#include "quantized_matmul.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"
#include "velomind/dtype.h"
#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/quant.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace velomind::backend::cpu {

namespace {

inline auto sign_extend_4bit(std::uint8_t nibble) noexcept -> std::int8_t {
    if (nibble & 0x08) {
        return static_cast<std::int8_t>(nibble | 0xF0);
    }
    return static_cast<std::int8_t>(nibble & 0x0F);
}

inline bool cpu_has_avx2_fma() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return supported;
#else
    return false;
#endif
}

} // namespace

template <typename TAct, typename TScale, typename TOut>
void quantized_matmul_scalar_reference(
    const TAct*         a,
    const std::int8_t*  b_q,
    const TScale*       scale,
    TOut*               c,
    std::size_t         m,
    std::size_t         k,
    std::size_t         n,
    std::size_t         a_row_stride,
    std::size_t         a_col_stride,
    std::size_t         b_row_stride,
    std::size_t         b_col_stride,
    std::size_t         c_row_stride,
    std::size_t         c_col_stride,
    QuantType           quant_type,
    int                 block_size)
{
    if (quant_type == QuantType::Int4) {
        const auto* b_packed = reinterpret_cast<const std::uint8_t*>(b_q);
        const std::size_t bs = (block_size > 0) ? static_cast<std::size_t>(block_size) : 32;
        const std::size_t num_blocks = (k + bs - 1) / bs;

        for (std::size_t i = 0; i < m; ++i) {
            const TAct* a_row = a + i * a_row_stride;
            TOut*       c_row = c + i * c_row_stride;

            for (std::size_t j = 0; j < n; ++j) {
                float total_acc = 0.0f;
                for (std::size_t b = 0; b < num_blocks; ++b) {
                    const std::size_t p_start = b * bs;
                    const std::size_t p_end   = std::min(k, p_start + bs);
                    const float s = static_cast<float>(scale[b * n + j]);

                    float block_acc = 0.0f;
                    for (std::size_t p = p_start; p < p_end; ++p) {
                        const std::uint8_t packed = b_packed[(p / 2) * b_row_stride + j * b_col_stride];
                        const std::uint8_t nibble = (p % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                        const auto w_val = static_cast<float>(sign_extend_4bit(nibble));
                        const auto a_val = static_cast<float>(a_row[p * a_col_stride]);
                        block_acc += a_val * w_val;
                    }
                    total_acc += block_acc * s;
                }
                c_row[j * c_col_stride] = static_cast<TOut>(total_acc);
            }
        }
        return;
    }

    if (block_size > 0) {
        const auto bs = static_cast<std::size_t>(block_size);
        const std::size_t num_blocks = (k + bs - 1) / bs;

        for (std::size_t i = 0; i < m; ++i) {
            const TAct* a_row = a + i * a_row_stride;
            TOut*       c_row = c + i * c_row_stride;

            for (std::size_t j = 0; j < n; ++j) {
                float total_acc = 0.0f;
                for (std::size_t b = 0; b < num_blocks; ++b) {
                    const std::size_t p_start = b * bs;
                    const std::size_t p_end   = std::min(k, p_start + bs);
                    const float s = static_cast<float>(scale[b * n + j]);

                    float block_acc = 0.0f;
                    for (std::size_t p = p_start; p < p_end; ++p) {
                        const auto a_val = static_cast<float>(a_row[p * a_col_stride]);
                        const auto w_val = static_cast<float>(b_q[p * b_row_stride + j * b_col_stride]);
                        block_acc += a_val * w_val;
                    }
                    total_acc += block_acc * s;
                }
                c_row[j * c_col_stride] = static_cast<TOut>(total_acc);
            }
        }
    } else {
        // 对称通道量化：列内累加后乘 scale
        for (std::size_t i = 0; i < m; ++i) {
            const TAct* a_row = a + i * a_row_stride;
            TOut*       c_row = c + i * c_row_stride;

            for (std::size_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (std::size_t p = 0; p < k; ++p) {
                    const auto a_val = static_cast<float>(a_row[p * a_col_stride]);
                    const auto w_val = static_cast<float>(b_q[p * b_row_stride + j * b_col_stride]);
                    acc += a_val * w_val;
                }
                const float s = static_cast<float>(scale[j]);
                c_row[j * c_col_stride] = static_cast<TOut>(acc * s);
            }
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void decode_stream_int8_avx2(
    const float*        a,
    const std::int8_t*  b_q,
    const float*        scale,
    float*              c,
    std::size_t         k,
    std::size_t         n,
    std::size_t         b_row_stride)
{
    constexpr std::size_t CHUNK_N = 64;

    for (std::size_t n0 = 0; n0 < n; n0 += CHUNK_N) {
        const std::size_t cur_chunk = std::min(n - n0, CHUNK_N);

        alignas(32) float accum[CHUNK_N] = {0.0f};

        for (std::size_t p = 0; p < k; ++p) {
            const float a_val = a[p];
            const __m256 va = _mm256_set1_ps(a_val);
            const std::int8_t* b_row = b_q + p * b_row_stride + n0;

            std::size_t j = 0;
            for (; j + 7 < cur_chunk; j += 8) {
                const __m128i raw8 = _mm_loadu_si64(b_row + j);
                const __m256i i32_vec = _mm256_cvtepi8_epi32(raw8);
                const __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
                __m256 vacc = _mm256_loadu_ps(accum + j);
                vacc = _mm256_fmadd_ps(va, f32_vec, vacc);
                _mm256_storeu_ps(accum + j, vacc);
            }
            for (; j < cur_chunk; ++j) {
                accum[j] += a_val * static_cast<float>(b_row[j]);
            }
        }

        // 按通道乘尺度写回
        for (std::size_t j = 0; j < cur_chunk; ++j) {
            c[n0 + j] = accum[j] * scale[n0 + j];
        }
    }
}
#pragma GCC pop_options
#endif

bool quantized_matmul_decode_f32(
    const float*        a,
    const std::int8_t*  b_q,
    const float*        scale,
    float*              c,
    std::size_t         k,
    std::size_t         n,
    std::size_t         a_stride,
    std::size_t         b_row_stride,
    std::size_t         b_col_stride,
    QuantType           quant_type,
    int                 block_size)
{
    if (quant_type != QuantType::Int8 || block_size != 0) {
        return false;
    }
    if (a_stride != 1 || b_col_stride != 1) {
        return false;
    }

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_avx2_fma()) {
        decode_stream_int8_avx2(a, b_q, scale, c, k, n, b_row_stride);
        return true;
    }
#endif

    // 纯 C++ 连续流式分块回退
    constexpr std::size_t CHUNK_N = 64;
    for (std::size_t n0 = 0; n0 < n; n0 += CHUNK_N) {
        const std::size_t cur_chunk = std::min(n - n0, CHUNK_N);
        std::vector<float> accum(cur_chunk, 0.0f);

        for (std::size_t p = 0; p < k; ++p) {
            const float a_val = a[p];
            const std::int8_t* b_row = b_q + p * b_row_stride + n0;
            for (std::size_t j = 0; j < cur_chunk; ++j) {
                accum[j] += a_val * static_cast<float>(b_row[j]);
            }
        }
        for (std::size_t j = 0; j < cur_chunk; ++j) {
            c[n0 + j] = accum[j] * scale[n0 + j];
        }
    }
    return true;
}

void quantized_matmul_2d_f32(
    const float*        a,
    const std::int8_t*  b_q,
    const float*        scale,
    float*              c,
    std::size_t         m,
    std::size_t         k,
    std::size_t         n,
    std::size_t         a_row_stride,
    std::size_t         a_col_stride,
    std::size_t         b_row_stride,
    std::size_t         b_col_stride,
    std::size_t         c_row_stride,
    std::size_t         c_col_stride,
    QuantType           quant_type,
    int                 block_size)
{
    if (m == 1 && c_col_stride == 1) {
        if (quantized_matmul_decode_f32(a, b_q, scale, c, k, n,
                                        a_col_stride, b_row_stride, b_col_stride,
                                        quant_type, block_size)) {
            return;
        }
    }

    quantized_matmul_scalar_reference(a, b_q, scale, c, m, k, n,
                                     a_row_stride, a_col_stride,
                                     b_row_stride, b_col_stride,
                                     c_row_stride, c_col_stride,
                                     quant_type, block_size);
}

namespace {

template <typename TAct, typename TScale, typename TOut>
void quantized_matmul_impl(
    const pConstTensorStorage* in,
    const pTensorStorage*      out,
    const void*                attrs)
{
    const auto* act_stor   = in[0];
    const auto* weight_stor = in[1];
    const auto* scale_stor  = in[2];
    auto*       out_stor    = out[0];

    if (!act_stor || !weight_stor || !scale_stor || !out_stor) {
        throw std::runtime_error("quantized_matmul: null storage");
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
    const std::size_t b_rank = b_shape.size();
    const std::size_t c_rank = c_shape.size();

    const std::size_t m = static_cast<std::size_t>(a_shape[a_rank - 2]);
    const std::size_t k = static_cast<std::size_t>(a_shape[a_rank - 1]);
    const std::size_t n = static_cast<std::size_t>(b_shape[1]);

    const auto a_strides = act_stor->effective_strides();
    const auto b_strides = weight_stor->effective_strides();
    const auto c_strides = out_stor->effective_strides();

    const std::size_t a_row_stride = a_strides[a_rank - 2];
    const std::size_t a_col_stride = a_strides[a_rank - 1];
    const std::size_t b_row_stride = b_strides[b_rank - 2];
    const std::size_t b_col_stride = b_strides[b_rank - 1];
    const std::size_t c_row_stride = c_strides[c_rank - 2];
    const std::size_t c_col_stride = c_strides[c_rank - 1];

    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < c_rank; ++i) {
        batch *= static_cast<std::size_t>(c_shape[i]);
    }

    std::vector<std::size_t> batch_coords(c_rank > 2 ? c_rank - 2 : 0, 0);

    for (std::size_t bi = 0; bi < batch; ++bi) {
        std::size_t a_batch_offset = 0;
        std::size_t c_batch_offset = 0;

        if (c_rank > 2) {
            std::size_t rem = bi;
            for (std::size_t d = c_rank - 2; d-- > 0;) {
                const std::size_t dim_sz = static_cast<std::size_t>(c_shape[d]);
                batch_coords[d] = rem % dim_sz;
                rem /= dim_sz;
            }
            for (std::size_t d = 0; d < c_rank - 2; ++d) {
                c_batch_offset += batch_coords[d] * c_strides[d];
                if (a_rank > 2 && d < a_rank - 2) {
                    a_batch_offset += batch_coords[d] * a_strides[d];
                }
            }
        }

        const TAct* a_mat = a + a_batch_offset;
        TOut*       c_mat = c + c_batch_offset;

        if constexpr (std::is_same_v<TAct, float> && std::is_same_v<TScale, float> && std::is_same_v<TOut, float>) {
            quantized_matmul_2d_f32(a_mat, b_q, s, c_mat, m, k, n,
                                    a_row_stride, a_col_stride,
                                    b_row_stride, b_col_stride,
                                    c_row_stride, c_col_stride,
                                    attr.quant_type, attr.block_size);
        } else {
            quantized_matmul_scalar_reference(a_mat, b_q, s, c_mat, m, k, n,
                                             a_row_stride, a_col_stride,
                                             b_row_stride, b_col_stride,
                                             c_row_stride, c_col_stride,
                                             attr.quant_type, attr.block_size);
        }
    }
}

// 静态注册 CPU 后端算子内核
static ::velomind::internal::KernelRegistrar _velomind_kr_qmatmul_f32_f32(
    DeviceType::CPU, Op::QuantizedMatMul,
    ::velomind::internal::KernelDtypeKey{
        {DataType::Float32, DataType::Int8, DataType::Float32}, 3, DataType::Float32},
    &quantized_matmul_impl<float, float, float>);

static ::velomind::internal::KernelRegistrar _velomind_kr_qmatmul_f16_f16(
    DeviceType::CPU, Op::QuantizedMatMul,
    ::velomind::internal::KernelDtypeKey{
        {DataType::Float16, DataType::Int8, DataType::Float16}, 3, DataType::Float16},
    &quantized_matmul_impl<float16_t, float16_t, float16_t>);

static ::velomind::internal::KernelRegistrar _velomind_kr_qmatmul_f16_f32(
    DeviceType::CPU, Op::QuantizedMatMul,
    ::velomind::internal::KernelDtypeKey{
        {DataType::Float16, DataType::Int8, DataType::Float32}, 3, DataType::Float16},
    &quantized_matmul_impl<float16_t, float, float16_t>);

static ::velomind::internal::KernelRegistrar _velomind_kr_qmatmul_bf16_bf16(
    DeviceType::CPU, Op::QuantizedMatMul,
    ::velomind::internal::KernelDtypeKey{
        {DataType::BFloat16, DataType::Int8, DataType::BFloat16}, 3, DataType::BFloat16},
    &quantized_matmul_impl<bfloat16_t, bfloat16_t, bfloat16_t>);

static ::velomind::internal::KernelRegistrar _velomind_kr_qmatmul_bf16_f32(
    DeviceType::CPU, Op::QuantizedMatMul,
    ::velomind::internal::KernelDtypeKey{
        {DataType::BFloat16, DataType::Int8, DataType::Float32}, 3, DataType::BFloat16},
    &quantized_matmul_impl<bfloat16_t, float, bfloat16_t>);

} // namespace

VELOMIND_REGISTER_SIGNATURE(Op::QuantizedMatMul, ::velomind::detail::signature_quantized_matmul)

} // namespace velomind::backend::cpu
