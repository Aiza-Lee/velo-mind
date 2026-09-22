#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "velomind/dtype.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind_core_export.h"

namespace velomind {

// 量化类型定义：支持 8 位整型对称量化与 4 位整型分块量化
enum class QuantType : std::uint8_t {
    None = 0,
    Int8 = 1,
    Int4 = 2,
};

constexpr const char* quant_type_name(QuantType type) noexcept {
    switch (type) {
        case QuantType::None: return "None";
        case QuantType::Int8: return "Int8";
        case QuantType::Int4: return "Int4";
    }
    return "Unknown";
}

// 量化数值误差与保真度评估指标
struct QuantizationMetrics {
    float max_diff   = 0.0f;
    float rmse       = 0.0f;
    float snr_db     = 0.0f;
    float cosine_sim = 1.0f;
};

// 计算原始浮点数据与量化重构数据的数值误差与余弦相似度
VELOMIND_CORE_EXPORT auto compute_quantization_metrics(
    const float* original,
    const float* reconstructed,
    std::size_t  count) -> QuantizationMetrics;

// 对称通道级 INT8 量化：针对矩阵 [rows, cols]，以每列（输出通道）为单位计算缩放系数并对称映射到 [-127, 127]
VELOMIND_CORE_EXPORT void quantize_weights_int8_per_channel(
    const float* src,
    std::int8_t* dst_q,
    float*       dst_scale,
    std::size_t  rows,
    std::size_t  cols);

// 对称通道级 INT8 反量化重构
VELOMIND_CORE_EXPORT void dequantize_weights_int8_per_channel(
    const std::int8_t* src_q,
    const float*       scale,
    float*             dst,
    std::size_t        rows,
    std::size_t        cols);

// 分块对称 INT4 量化：针对矩阵 [rows, cols]，沿列每 block_size（默认 32）个元素映射到 [-8, 7] 并打包存储（2 权重复用 1 字节）
VELOMIND_CORE_EXPORT void quantize_weights_int4_block(
    const float*  src,
    std::uint8_t* dst_packed,
    float*        dst_scale,
    std::size_t   rows,
    std::size_t   cols,
    std::size_t   block_size = 32);

// 分块对称 INT4 反量化重构
VELOMIND_CORE_EXPORT void dequantize_weights_int4_block(
    const std::uint8_t* src_packed,
    const float*        scale,
    float*              dst,
    std::size_t         rows,
    std::size_t         cols,
    std::size_t         block_size = 32);

// TensorStorage 封装量化：针对二维连续张量返回 (量化权重存储, 尺度张量存储)
VELOMIND_CORE_EXPORT auto quantize_storage_int8(
    const TensorStorage& src_storage,
    DeviceType           device = DeviceType::CPU)
    -> std::pair<std::shared_ptr<TensorStorage>, std::shared_ptr<TensorStorage>>;

VELOMIND_CORE_EXPORT auto quantize_storage_int4(
    const TensorStorage& src_storage,
    std::size_t          block_size = 32,
    DeviceType           device = DeviceType::CPU)
    -> std::pair<std::shared_ptr<TensorStorage>, std::shared_ptr<TensorStorage>>;

} // namespace velomind
