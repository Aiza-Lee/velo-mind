#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "velomind/dtype.h"
#include "velomind/ops.h"
#include "velomind/quant.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind_core_export.h"

namespace velomind::backend::cpu {

// 标量参考实现：支持任意批次维度、步长、类型及 INT8/INT4 量化模式
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
    int                 block_size);

// M=1 自回归解码单 token 专用流式内核（AVX2+FMA 向量化）
VELOMIND_CORE_EXPORT bool quantized_matmul_decode_f32(
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
    int                 block_size);

// 通用 2D Float32 量化 GEMM/GEMV 调度入口
VELOMIND_CORE_EXPORT void quantized_matmul_2d_f32(
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
    int                 block_size);

} // namespace velomind::backend::cpu
