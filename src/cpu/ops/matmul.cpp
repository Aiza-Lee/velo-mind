#include "matmul.h"

#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace velomind::backend::cpu {

namespace {

inline bool cpu_has_avx2_fma() {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return supported;
#else
    return false;
#endif
}

inline float dot_product_portable(const float* a, const float* b, std::size_t k) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    std::size_t p = 0;
    for (; p + 3 < k; p += 4) {
        sum0 += a[p + 0] * b[p + 0];
        sum1 += a[p + 1] * b[p + 1];
        sum2 += a[p + 2] * b[p + 2];
        sum3 += a[p + 3] * b[p + 3];
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; p < k; ++p) {
        sum += a[p] * b[p];
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
inline float dot_product_avx2(const float* a, const float* b, std::size_t k) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    std::size_t p = 0;
    for (; p + 31 < k; p += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8), _mm256_loadu_ps(b + p + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16), _mm256_loadu_ps(b + p + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24), _mm256_loadu_ps(b + p + 24), acc3);
    }
    for (; p + 7 < k; p += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    float tmp[8];
    _mm256_storeu_ps(tmp, acc0);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; p < k; ++p) {
        sum += a[p] * b[p];
    }
    return sum;
}

void small_m_stream_rowmajor_avx2(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t b_row_stride, std::size_t c_row_stride)
{
    constexpr std::size_t BLOCK_N = 512;

    if (m == 1) {
        for (std::size_t n0 = 0; n0 < n; n0 += BLOCK_N) {
            const std::size_t n_chunk = std::min(n - n0, BLOCK_N);
            float* c_ptr = c + n0;
            std::memset(c_ptr, 0, n_chunk * sizeof(float));

            for (std::size_t p = 0; p < k; ++p) {
                const float a_val = a[p];
                const __m256 av = _mm256_set1_ps(a_val);
                const float* b_row = b + p * b_row_stride + n0;

                std::size_t j = 0;
                for (; j + 31 < n_chunk; j += 32) {
                    __m256 b0 = _mm256_loadu_ps(b_row + j);
                    __m256 b1 = _mm256_loadu_ps(b_row + j + 8);
                    __m256 b2 = _mm256_loadu_ps(b_row + j + 16);
                    __m256 b3 = _mm256_loadu_ps(b_row + j + 24);

                    __m256 cv0 = _mm256_loadu_ps(c_ptr + j);
                    __m256 cv1 = _mm256_loadu_ps(c_ptr + j + 8);
                    __m256 cv2 = _mm256_loadu_ps(c_ptr + j + 16);
                    __m256 cv3 = _mm256_loadu_ps(c_ptr + j + 24);

                    cv0 = _mm256_fmadd_ps(av, b0, cv0);
                    cv1 = _mm256_fmadd_ps(av, b1, cv1);
                    cv2 = _mm256_fmadd_ps(av, b2, cv2);
                    cv3 = _mm256_fmadd_ps(av, b3, cv3);

                    _mm256_storeu_ps(c_ptr + j, cv0);
                    _mm256_storeu_ps(c_ptr + j + 8, cv1);
                    _mm256_storeu_ps(c_ptr + j + 16, cv2);
                    _mm256_storeu_ps(c_ptr + j + 24, cv3);
                }
                for (; j + 7 < n_chunk; j += 8) {
                    __m256 bv = _mm256_loadu_ps(b_row + j);
                    __m256 cv = _mm256_loadu_ps(c_ptr + j);
                    cv = _mm256_fmadd_ps(av, bv, cv);
                    _mm256_storeu_ps(c_ptr + j, cv);
                }
                for (; j < n_chunk; ++j) {
                    c_ptr[j] += a_val * b_row[j];
                }
            }
        }
        return;
    }

    for (std::size_t n0 = 0; n0 < n; n0 += BLOCK_N) {
        const std::size_t n_chunk = std::min(n - n0, BLOCK_N);
        for (std::size_t i = 0; i < m; ++i) {
            std::memset(c + i * c_row_stride + n0, 0, n_chunk * sizeof(float));
        }

        for (std::size_t p = 0; p < k; ++p) {
            const float* b_row = b + p * b_row_stride + n0;

            __m256 av[4];
            for (std::size_t i = 0; i < m; ++i) {
                av[i] = _mm256_set1_ps(a[i * a_row_stride + p]);
            }

            std::size_t j = 0;
            for (; j + 7 < n_chunk; j += 8) {
                const __m256 bv = _mm256_loadu_ps(b_row + j);
                for (std::size_t i = 0; i < m; ++i) {
                    float* c_ptr = c + i * c_row_stride + n0 + j;
                    __m256 cv = _mm256_loadu_ps(c_ptr);
                    cv = _mm256_fmadd_ps(av[i], bv, cv);
                    _mm256_storeu_ps(c_ptr, cv);
                }
            }
            for (; j < n_chunk; ++j) {
                const float b_val = b_row[j];
                for (std::size_t i = 0; i < m; ++i) {
                    c[i * c_row_stride + n0 + j] += a[i * a_row_stride + p] * b_val;
                }
            }
        }
    }
}

void blocked_packed_compute_avx2(
    const float* a, const float* b_packed, float* c,
    std::size_t m_len, std::size_t k_len, std::size_t n_len,
    std::size_t a_row_stride, std::size_t c_row_stride,
    std::size_t m0, std::size_t k0, std::size_t n0)
{
    std::size_t b_panel_offset = 0;
    for (std::size_t nb = 0; nb < n_len; nb += 16) {
        const std::size_t nr = std::min(n_len - nb, std::size_t(16));
        const float* b_p = b_packed + b_panel_offset;
        b_panel_offset += k_len * 16;

        std::size_t i = 0;
        for (; i + 3 < m_len; i += 4) {
            const std::size_t row0 = m0 + i;
            const std::size_t row1 = row0 + 1;
            const std::size_t row2 = row0 + 2;
            const std::size_t row3 = row0 + 3;

            float* c0 = c + row0 * c_row_stride + n0 + nb;
            float* c1 = c + row1 * c_row_stride + n0 + nb;
            float* c2 = c + row2 * c_row_stride + n0 + nb;
            float* c3 = c + row3 * c_row_stride + n0 + nb;

            __m256 cv00, cv01, cv10, cv11, cv20, cv21, cv30, cv31;
            if (nr == 16) {
                cv00 = _mm256_loadu_ps(c0);
                cv01 = _mm256_loadu_ps(c0 + 8);
                cv10 = _mm256_loadu_ps(c1);
                cv11 = _mm256_loadu_ps(c1 + 8);
                cv20 = _mm256_loadu_ps(c2);
                cv21 = _mm256_loadu_ps(c2 + 8);
                cv30 = _mm256_loadu_ps(c3);
                cv31 = _mm256_loadu_ps(c3 + 8);
            } else {
                float tmp0[16] = {0}, tmp1[16] = {0}, tmp2[16] = {0}, tmp3[16] = {0};
                for (std::size_t j = 0; j < nr; ++j) {
                    tmp0[j] = c0[j];
                    tmp1[j] = c1[j];
                    tmp2[j] = c2[j];
                    tmp3[j] = c3[j];
                }
                cv00 = _mm256_loadu_ps(tmp0);
                cv01 = _mm256_loadu_ps(tmp0 + 8);
                cv10 = _mm256_loadu_ps(tmp1);
                cv11 = _mm256_loadu_ps(tmp1 + 8);
                cv20 = _mm256_loadu_ps(tmp2);
                cv21 = _mm256_loadu_ps(tmp2 + 8);
                cv30 = _mm256_loadu_ps(tmp3);
                cv31 = _mm256_loadu_ps(tmp3 + 8);
            }

            for (std::size_t p = 0; p < k_len; ++p) {
                const __m256 bp0 = _mm256_loadu_ps(b_p + p * 16);
                const __m256 bp1 = _mm256_loadu_ps(b_p + p * 16 + 8);

                const __m256 a0 = _mm256_set1_ps(a[row0 * a_row_stride + k0 + p]);
                const __m256 a1 = _mm256_set1_ps(a[row1 * a_row_stride + k0 + p]);
                const __m256 a2 = _mm256_set1_ps(a[row2 * a_row_stride + k0 + p]);
                const __m256 a3 = _mm256_set1_ps(a[row3 * a_row_stride + k0 + p]);

                cv00 = _mm256_fmadd_ps(a0, bp0, cv00);
                cv01 = _mm256_fmadd_ps(a0, bp1, cv01);
                cv10 = _mm256_fmadd_ps(a1, bp0, cv10);
                cv11 = _mm256_fmadd_ps(a1, bp1, cv11);
                cv20 = _mm256_fmadd_ps(a2, bp0, cv20);
                cv21 = _mm256_fmadd_ps(a2, bp1, cv21);
                cv30 = _mm256_fmadd_ps(a3, bp0, cv30);
                cv31 = _mm256_fmadd_ps(a3, bp1, cv31);
            }

            if (nr == 16) {
                _mm256_storeu_ps(c0, cv00);
                _mm256_storeu_ps(c0 + 8, cv01);
                _mm256_storeu_ps(c1, cv10);
                _mm256_storeu_ps(c1 + 8, cv11);
                _mm256_storeu_ps(c2, cv20);
                _mm256_storeu_ps(c2 + 8, cv21);
                _mm256_storeu_ps(c3, cv30);
                _mm256_storeu_ps(c3 + 8, cv31);
            } else {
                float tmp[16];
                _mm256_storeu_ps(tmp, cv00);
                _mm256_storeu_ps(tmp + 8, cv01);
                for (std::size_t j = 0; j < nr; ++j) c0[j] = tmp[j];

                _mm256_storeu_ps(tmp, cv10);
                _mm256_storeu_ps(tmp + 8, cv11);
                for (std::size_t j = 0; j < nr; ++j) c1[j] = tmp[j];

                _mm256_storeu_ps(tmp, cv20);
                _mm256_storeu_ps(tmp + 8, cv21);
                for (std::size_t j = 0; j < nr; ++j) c2[j] = tmp[j];

                _mm256_storeu_ps(tmp, cv30);
                _mm256_storeu_ps(tmp + 8, cv31);
                for (std::size_t j = 0; j < nr; ++j) c3[j] = tmp[j];
            }
        }
        for (; i < m_len; ++i) {
            const std::size_t row = m0 + i;
            float* c_row = c + row * c_row_stride + n0 + nb;
            __m256 cv0, cv1;
            if (nr == 16) {
                cv0 = _mm256_loadu_ps(c_row);
                cv1 = _mm256_loadu_ps(c_row + 8);
            } else {
                float tmp[16] = {0};
                for (std::size_t j = 0; j < nr; ++j) tmp[j] = c_row[j];
                cv0 = _mm256_loadu_ps(tmp);
                cv1 = _mm256_loadu_ps(tmp + 8);
            }
            for (std::size_t p = 0; p < k_len; ++p) {
                const __m256 a_val = _mm256_set1_ps(a[row * a_row_stride + k0 + p]);
                const __m256 bp0 = _mm256_loadu_ps(b_p + p * 16);
                const __m256 bp1 = _mm256_loadu_ps(b_p + p * 16 + 8);
                cv0 = _mm256_fmadd_ps(a_val, bp0, cv0);
                cv1 = _mm256_fmadd_ps(a_val, bp1, cv1);
            }
            if (nr == 16) {
                _mm256_storeu_ps(c_row, cv0);
                _mm256_storeu_ps(c_row + 8, cv1);
            } else {
                float tmp[16];
                _mm256_storeu_ps(tmp, cv0);
                _mm256_storeu_ps(tmp + 8, cv1);
                for (std::size_t j = 0; j < nr; ++j) c_row[j] = tmp[j];
            }
        }
    }
}
#pragma GCC pop_options
#endif

void small_m_stream_rowmajor_portable(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t b_row_stride, std::size_t c_row_stride)
{
    constexpr std::size_t BLOCK_N = 512;

    for (std::size_t n0 = 0; n0 < n; n0 += BLOCK_N) {
        const std::size_t n_chunk = std::min(n - n0, BLOCK_N);
        for (std::size_t i = 0; i < m; ++i) {
            std::memset(c + i * c_row_stride + n0, 0, n_chunk * sizeof(float));
        }

        for (std::size_t p = 0; p < k; ++p) {
            const float* b_row = b + p * b_row_stride + n0;
            for (std::size_t i = 0; i < m; ++i) {
                const float a_val = a[i * a_row_stride + p];
                float* c_row = c + i * c_row_stride + n0;
                #pragma GCC ivdep
                for (std::size_t j = 0; j < n_chunk; ++j) {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}

void blocked_packed_compute_portable(
    const float* a, const float* b_packed, float* c,
    std::size_t m_len, std::size_t k_len, std::size_t n_len,
    std::size_t a_row_stride, std::size_t c_row_stride,
    std::size_t m0, std::size_t k0, std::size_t n0)
{
    std::size_t b_panel_offset = 0;
    for (std::size_t nb = 0; nb < n_len; nb += 16) {
        const std::size_t nr = std::min(n_len - nb, std::size_t(16));
        const float* b_p = b_packed + b_panel_offset;
        b_panel_offset += k_len * 16;

        for (std::size_t i = 0; i < m_len; ++i) {
            const std::size_t row = m0 + i;
            float* c_row = c + row * c_row_stride + n0 + nb;
            for (std::size_t p = 0; p < k_len; ++p) {
                const float a_val = a[row * a_row_stride + k0 + p];
                const float* b_row = b_p + p * 16;
                #pragma GCC ivdep
                for (std::size_t j = 0; j < nr; ++j) {
                    c_row[j] += a_val * b_row[j];
                }
            }
        }
    }
}

} // namespace

void pack_b_panel(
    const float* b, float* b_packed,
    std::size_t k_start, std::size_t k_len,
    std::size_t n_start, std::size_t n_len,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t nr)
{
    std::size_t out_idx = 0;
    for (std::size_t n0 = 0; n0 < n_len; n0 += nr) {
        const std::size_t cur_nr = std::min(n_len - n0, nr);
        for (std::size_t p = 0; p < k_len; ++p) {
            const float* b_row = b + (k_start + p) * b_row_stride + (n_start + n0) * b_col_stride;
            if (cur_nr == nr && b_col_stride == 1) {
                std::memcpy(b_packed + out_idx, b_row, nr * sizeof(float));
            } else {
                for (std::size_t j = 0; j < cur_nr; ++j) {
                    b_packed[out_idx + j] = b_row[j * b_col_stride];
                }
                for (std::size_t j = cur_nr; j < nr; ++j) {
                    b_packed[out_idx + j] = 0.0f;
                }
            }
            out_idx += nr;
        }
    }
}

bool matmul_small_m_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride)
{
    if (m > 4 || c_col_stride != 1 || a_col_stride != 1) {
        return false;
    }

    // B 为行优先连续
    if (b_col_stride == 1) {
#if defined(__x86_64__) || defined(_M_X64)
        if (cpu_has_avx2_fma()) {
            small_m_stream_rowmajor_avx2(a, b, c, m, k, n,
                                         a_row_stride, b_row_stride, c_row_stride);
            return true;
        }
#endif
        small_m_stream_rowmajor_portable(a, b, c, m, k, n,
                                         a_row_stride, b_row_stride, c_row_stride);
        return true;
    }

    // B 为列优先连续
    if (b_row_stride == 1) {
        const bool use_avx2 = cpu_has_avx2_fma();
        for (std::size_t j = 0; j < n; ++j) {
            const float* b_col = b + j * b_col_stride;
            for (std::size_t i = 0; i < m; ++i) {
                const float* a_row = a + i * a_row_stride;
#if defined(__x86_64__) || defined(_M_X64)
                if (use_avx2) {
                    c[i * c_row_stride + j] = dot_product_avx2(a_row, b_col, k);
                    continue;
                }
#endif
                c[i * c_row_stride + j] = dot_product_portable(a_row, b_col, k);
            }
        }
        return true;
    }

    return false;
}

bool matmul_blocked_packed_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride)
{
    if (c_col_stride != 1 || a_col_stride != 1) {
        return false;
    }

    constexpr std::size_t KC = 128;
    constexpr std::size_t NC = 128;
    constexpr std::size_t MC = 64;

    std::vector<float> b_packed(KC * NC);

    for (std::size_t i = 0; i < m; ++i) {
        std::memset(c + i * c_row_stride, 0, n * sizeof(float));
    }

    const bool use_avx2 = cpu_has_avx2_fma();

    for (std::size_t k0 = 0; k0 < k; k0 += KC) {
        const std::size_t k_len = std::min(k - k0, KC);

        for (std::size_t n0 = 0; n0 < n; n0 += NC) {
            const std::size_t n_len = std::min(n - n0, NC);

            pack_b_panel(b, b_packed.data(), k0, k_len, n0, n_len, b_row_stride, b_col_stride, 16);

            for (std::size_t m0 = 0; m0 < m; m0 += MC) {
                const std::size_t m_len = std::min(m - m0, MC);
#if defined(__x86_64__) || defined(_M_X64)
                if (use_avx2) {
                    blocked_packed_compute_avx2(a, b_packed.data(), c,
                                                m_len, k_len, n_len,
                                                a_row_stride, c_row_stride,
                                                m0, k0, n0);
                    continue;
                }
#endif
                blocked_packed_compute_portable(a, b_packed.data(), c,
                                                m_len, k_len, n_len,
                                                a_row_stride, c_row_stride,
                                                m0, k0, n0);
            }
        }
    }

    return true;
}

void matmul_2d_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride)
{
    if (m <= 4) {
        if (matmul_small_m_f32(a, b, c, m, k, n,
                               a_row_stride, a_col_stride,
                               b_row_stride, b_col_stride,
                               c_row_stride, c_col_stride)) {
            return;
        }
    } else {
        if (matmul_blocked_packed_f32(a, b, c, m, k, n,
                                     a_row_stride, a_col_stride,
                                     b_row_stride, b_col_stride,
                                     c_row_stride, c_col_stride)) {
            return;
        }
    }

    // 标量参考兜底
    matmul_scalar_reference(a, b, c, m, k, n,
                            a_row_stride, a_col_stride,
                            b_row_stride, b_col_stride,
                            c_row_stride, c_col_stride);
}

namespace {

template <typename T1, typename T2, typename T3>
void matmul_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T1* a = static_cast<const T1*>(in[0]->data);
    const T2* b = static_cast<const T2*>(in[1]->data);
    T3*       c = static_cast<T3*>(out[0]->data);

    const auto& a_shape = in[0]->shape;
    const auto& b_shape = in[1]->shape;
    const auto& c_shape = out[0]->shape;

    const std::size_t a_rank = a_shape.size();
    const std::size_t b_rank = b_shape.size();
    const std::size_t c_rank = c_shape.size();

    if (a_rank < 2 || b_rank < 2 || c_rank < 2) {
        throw std::runtime_error("velomind::MatMul: rank must be >= 2");
    }

    const std::size_t m = static_cast<std::size_t>(a_shape[a_rank - 2]);
    const std::size_t k = static_cast<std::size_t>(a_shape[a_rank - 1]);
    const std::size_t n = static_cast<std::size_t>(b_shape[b_rank - 1]);
    if (static_cast<std::size_t>(b_shape[b_rank - 2]) != k) {
        throw std::runtime_error("velomind::MatMul: K dimension mismatch");
    }
    if (static_cast<std::size_t>(c_shape[c_rank - 2]) != m ||
        static_cast<std::size_t>(c_shape[c_rank - 1]) != n) {
        throw std::runtime_error("velomind::MatMul: output shape mismatch");
    }

    const auto a_strides = in[0]->effective_strides();
    const auto b_strides = in[1]->effective_strides();
    const auto c_strides = out[0]->effective_strides();

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
        std::size_t b_batch_offset = 0;
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
                if (b_rank > 2 && d < b_rank - 2) {
                    b_batch_offset += batch_coords[d] * b_strides[d];
                }
            }
        }

        const T1* a_mat = a + a_batch_offset;
        const T2* b_mat = b + b_batch_offset;
        T3*       c_mat = c + c_batch_offset;

        if constexpr (std::is_same_v<T1, float> && std::is_same_v<T2, float> && std::is_same_v<T3, float>) {
            matmul_2d_f32(a_mat, b_mat, c_mat, m, k, n,
                          a_row_stride, a_col_stride,
                          b_row_stride, b_col_stride,
                          c_row_stride, c_col_stride);
        } else {
            matmul_scalar_reference(a_mat, b_mat, c_mat, m, k, n,
                                    a_row_stride, a_col_stride,
                                    b_row_stride, b_col_stride,
                                    c_row_stride, c_col_stride);
        }
    }
}

VELOMIND_REGISTER_BINARY_SAME(DeviceType::CPU, Op::MatMul, matmul_impl);

VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::Float16, DataType::Float16, DataType::Float16,
                                matmul_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::Float16, DataType::Float16, DataType::Float32,
                                matmul_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::Float32, DataType::Float16, DataType::Float32,
                                matmul_impl);

VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::BFloat16, DataType::BFloat16, DataType::BFloat16,
                                matmul_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::BFloat16, DataType::BFloat16, DataType::Float32,
                                matmul_impl);
VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::MatMul,
                                DataType::Float32, DataType::BFloat16, DataType::Float32,
                                matmul_impl);

} // namespace

VELOMIND_REGISTER_SIGNATURE(Op::MatMul, ::velomind::detail::signature_matmul)

} // namespace velomind::backend::cpu
