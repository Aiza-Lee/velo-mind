#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "velomind_core_export.h"

namespace velomind::backend::cpu {

// 标量参考实现：通用步长与类型逐元素计算，作为基准与复杂步长/非 Float32 回退路径。
template <typename T1, typename T2, typename T3>
void matmul_scalar_reference(
    const T1* a, const T2* b, T3* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride)
{
    using AccType = std::conditional_t<
        std::is_integral_v<T1> && std::is_integral_v<T2> && std::is_integral_v<T3>,
        T3,
        float
    >;
    for (std::size_t i = 0; i < m; ++i) {
        const T1* a_row = a + i * a_row_stride;
        T3*       c_row = c + i * c_row_stride;
        for (std::size_t j = 0; j < n; ++j) {
            AccType acc = AccType(0);
            const T2* b_col = b + j * b_col_stride;
            for (std::size_t p = 0; p < k; ++p) {
                acc += static_cast<AccType>(a_row[p * a_col_stride]) *
                       static_cast<AccType>(b_col[p * b_row_stride]);
            }
            c_row[j * c_col_stride] = static_cast<T3>(acc);
        }
    }
}

// 权重分块打包：将 B 的 [k_len, n_len] 子矩阵重组为面板宽度 nr 的连续微块，消除内部跨步访存。
VELOMIND_CORE_EXPORT void pack_b_panel(
    const float* b, float* b_packed,
    std::size_t k_start, std::size_t k_len,
    std::size_t n_start, std::size_t n_len,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t nr = 16);

// 小 M 专用路径（M <= 4）：针对 LLM 自回归 decode 单 token 及投机验证小 batch，消除 B 转置并最大化流式带宽。
VELOMIND_CORE_EXPORT bool matmul_small_m_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride);

// 分块与权重打包路径（M > 4）：针对 prefill 与全序列 GEMM，进行 L1/L2 缓存分块与权重微面板重排。
VELOMIND_CORE_EXPORT bool matmul_blocked_packed_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride);

// 2D 单批次 Float32 GEMM 统一分发入口。
VELOMIND_CORE_EXPORT void matmul_2d_f32(
    const float* a, const float* b, float* c,
    std::size_t m, std::size_t k, std::size_t n,
    std::size_t a_row_stride, std::size_t a_col_stride,
    std::size_t b_row_stride, std::size_t b_col_stride,
    std::size_t c_row_stride, std::size_t c_col_stride);

} // namespace velomind::backend::cpu
