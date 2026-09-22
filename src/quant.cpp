#include "velomind/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "velomind/dtype.h"
#include "velomind/tensor_storage.h"
#include "internal/registry/memory_transfer.h"

namespace velomind {

namespace {

inline auto sign_extend_4bit(std::uint8_t nibble) noexcept -> std::int8_t {
    if (nibble & 0x08) {
        return static_cast<std::int8_t>(nibble | 0xF0);
    }
    return static_cast<std::int8_t>(nibble & 0x0F);
}

} // namespace

auto compute_quantization_metrics(
    const float* original,
    const float* reconstructed,
    std::size_t  count) -> QuantizationMetrics
{
    if (count == 0 || !original || !reconstructed) {
        return QuantizationMetrics{};
    }

    float max_diff = 0.0f;
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    double sum_sq_recon = 0.0;
    double sum_prod = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double o = static_cast<double>(original[i]);
        const double r = static_cast<double>(reconstructed[i]);
        const double diff = std::abs(o - r);
        if (diff > max_diff) {
            max_diff = static_cast<float>(diff);
        }
        sum_sq_diff += diff * diff;
        sum_sq_orig += o * o;
        sum_sq_recon += r * r;
        sum_prod += o * r;
    }

    QuantizationMetrics metrics;
    metrics.max_diff = max_diff;
    metrics.rmse = static_cast<float>(std::sqrt(sum_sq_diff / static_cast<double>(count)));

    if (sum_sq_diff > 1e-15) {
        metrics.snr_db = static_cast<float>(10.0 * std::log10(sum_sq_orig / sum_sq_diff));
    } else {
        metrics.snr_db = 100.0f;
    }

    const double denom = std::sqrt(sum_sq_orig) * std::sqrt(sum_sq_recon);
    if (denom > 1e-15) {
        metrics.cosine_sim = static_cast<float>(sum_prod / denom);
    } else {
        metrics.cosine_sim = 1.0f;
    }

    return metrics;
}

void quantize_weights_int8_per_channel(
    const float* src,
    std::int8_t* dst_q,
    float*       dst_scale,
    std::size_t  rows,
    std::size_t  cols)
{
    if (!src || !dst_q || !dst_scale || rows == 0 || cols == 0) {
        return;
    }

    for (std::size_t c = 0; c < cols; ++c) {
        float max_abs = 0.0f;
        for (std::size_t r = 0; r < rows; ++r) {
            const float val = std::abs(src[r * cols + c]);
            if (val > max_abs) {
                max_abs = val;
            }
        }

        const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        dst_scale[c] = scale;
        const float inv_scale = 1.0f / scale;

        for (std::size_t r = 0; r < rows; ++r) {
            const float q_val = std::round(src[r * cols + c] * inv_scale);
            const float clamped = std::clamp(q_val, -127.0f, 127.0f);
            dst_q[r * cols + c] = static_cast<std::int8_t>(clamped);
        }
    }
}

void dequantize_weights_int8_per_channel(
    const std::int8_t* src_q,
    const float*       scale,
    float*             dst,
    std::size_t        rows,
    std::size_t        cols)
{
    if (!src_q || !scale || !dst || rows == 0 || cols == 0) {
        return;
    }

    for (std::size_t c = 0; c < cols; ++c) {
        const float s = scale[c];
        for (std::size_t r = 0; r < rows; ++r) {
            dst[r * cols + c] = static_cast<float>(src_q[r * cols + c]) * s;
        }
    }
}

void quantize_weights_int4_block(
    const float*  src,
    std::uint8_t* dst_packed,
    float*        dst_scale,
    std::size_t   rows,
    std::size_t   cols,
    std::size_t   block_size)
{
    if (!src || !dst_packed || !dst_scale || rows == 0 || cols == 0 || block_size == 0) {
        return;
    }

    const std::size_t num_blocks = (rows + block_size - 1) / block_size;
    const std::size_t packed_rows = (rows + 1) / 2;

    std::memset(dst_packed, 0, packed_rows * cols);

    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t r_start = b * block_size;
            const std::size_t r_end   = std::min(rows, r_start + block_size);

            float max_abs = 0.0f;
            for (std::size_t r = r_start; r < r_end; ++r) {
                const float val = std::abs(src[r * cols + c]);
                if (val > max_abs) {
                    max_abs = val;
                }
            }

            const float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 1.0f;
            dst_scale[b * cols + c] = scale;
            const float inv_scale = 1.0f / scale;

            for (std::size_t r = r_start; r < r_end; r += 2) {
                const float q0 = std::round(src[r * cols + c] * inv_scale);
                const auto c0 = static_cast<std::int8_t>(std::clamp(q0, -8.0f, 7.0f));
                const auto n0 = static_cast<std::uint8_t>(c0 & 0x0F);

                std::uint8_t n1 = 0;
                if (r + 1 < r_end) {
                    const float q1 = std::round(src[(r + 1) * cols + c] * inv_scale);
                    const auto c1 = static_cast<std::int8_t>(std::clamp(q1, -8.0f, 7.0f));
                    n1 = static_cast<std::uint8_t>(c1 & 0x0F);
                }

                dst_packed[(r / 2) * cols + c] = static_cast<std::uint8_t>(n0 | (n1 << 4));
            }
        }
    }
}

void dequantize_weights_int4_block(
    const std::uint8_t* src_packed,
    const float*        scale,
    float*              dst,
    std::size_t         rows,
    std::size_t         cols,
    std::size_t         block_size)
{
    if (!src_packed || !scale || !dst || rows == 0 || cols == 0 || block_size == 0) {
        return;
    }

    const std::size_t num_blocks = (rows + block_size - 1) / block_size;

    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t b = 0; b < num_blocks; ++b) {
            const std::size_t r_start = b * block_size;
            const std::size_t r_end   = std::min(rows, r_start + block_size);
            const float s = scale[b * cols + c];

            for (std::size_t r = r_start; r < r_end; r += 2) {
                const std::uint8_t packed = src_packed[(r / 2) * cols + c];
                const std::int8_t v0 = sign_extend_4bit(packed & 0x0F);
                dst[r * cols + c] = static_cast<float>(v0) * s;

                if (r + 1 < r_end) {
                    const std::int8_t v1 = sign_extend_4bit((packed >> 4) & 0x0F);
                    dst[(r + 1) * cols + c] = static_cast<float>(v1) * s;
                }
            }
        }
    }
}

auto quantize_storage_int8(
    const TensorStorage& src_storage,
    DeviceType           device)
    -> std::pair<std::shared_ptr<TensorStorage>, std::shared_ptr<TensorStorage>>
{
    if (src_storage.shape.size() != 2) {
        throw std::invalid_argument("quantize_storage_int8: only 2D weight matrices are supported");
    }

    const std::size_t rows = static_cast<std::size_t>(src_storage.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(src_storage.shape[1]);
    const std::size_t numel = rows * cols;

    std::vector<float> f32_buf;
    const float* f32_ptr = nullptr;

    if (src_storage.dtype == DataType::Float32) {
        f32_ptr = static_cast<const float*>(src_storage.data);
    } else {
        f32_buf.resize(numel);
        convert_dtype(src_storage.data, src_storage.dtype, f32_buf.data(), DataType::Float32, numel);
        f32_ptr = f32_buf.data();
    }

    std::vector<std::int8_t> q_host(numel);
    std::vector<float> scale_host(cols);

    quantize_weights_int8_per_channel(f32_ptr, q_host.data(), scale_host.data(), rows, cols);

    const std::size_t q_bytes = numel * sizeof(std::int8_t);
    const std::size_t scale_bytes = cols * sizeof(float);

    auto q_storage = TensorStorage::allocate(q_bytes, device);
    q_storage->shape = {static_cast<dim_t>(rows), static_cast<dim_t>(cols)};
    q_storage->dtype = DataType::Int8;
    q_storage->size_bytes = q_bytes;
    q_storage->capacity_bytes = q_bytes;
    q_storage->device = device;

    auto scale_storage = TensorStorage::allocate(scale_bytes, device);
    scale_storage->shape = {static_cast<dim_t>(cols)};
    scale_storage->dtype = DataType::Float32;
    scale_storage->size_bytes = scale_bytes;
    scale_storage->capacity_bytes = scale_bytes;
    scale_storage->device = device;

    if (device == DeviceType::CPU) {
        std::memcpy(q_storage->data, q_host.data(), q_bytes);
        std::memcpy(scale_storage->data, scale_host.data(), scale_bytes);
    } else {
        auto t = internal::get_memory_transfer(device);
        if (t.copy_h2d) {
            t.copy_h2d(q_storage->data, q_host.data(), q_bytes);
            t.copy_h2d(scale_storage->data, scale_host.data(), scale_bytes);
        } else {
            throw std::runtime_error("quantize_storage_int8: no H2D transfer for device");
        }
    }

    return {std::move(q_storage), std::move(scale_storage)};
}

auto quantize_storage_int4(
    const TensorStorage& src_storage,
    std::size_t          block_size,
    DeviceType           device)
    -> std::pair<std::shared_ptr<TensorStorage>, std::shared_ptr<TensorStorage>>
{
    if (src_storage.shape.size() != 2) {
        throw std::invalid_argument("quantize_storage_int4: only 2D weight matrices are supported");
    }
    if (block_size == 0) {
        throw std::invalid_argument("quantize_storage_int4: block_size must be positive");
    }

    const std::size_t rows = static_cast<std::size_t>(src_storage.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(src_storage.shape[1]);
    const std::size_t numel = rows * cols;
    const std::size_t num_blocks = (rows + block_size - 1) / block_size;
    const std::size_t packed_rows = (rows + 1) / 2;

    std::vector<float> f32_buf;
    const float* f32_ptr = nullptr;

    if (src_storage.dtype == DataType::Float32) {
        f32_ptr = static_cast<const float*>(src_storage.data);
    } else {
        f32_buf.resize(numel);
        convert_dtype(src_storage.data, src_storage.dtype, f32_buf.data(), DataType::Float32, numel);
        f32_ptr = f32_buf.data();
    }

    std::vector<std::uint8_t> packed_host(packed_rows * cols);
    std::vector<float> scale_host(num_blocks * cols);

    quantize_weights_int4_block(f32_ptr, packed_host.data(), scale_host.data(), rows, cols, block_size);

    const std::size_t q_bytes = packed_rows * cols;
    const std::size_t scale_bytes = num_blocks * cols * sizeof(float);

    auto q_storage = TensorStorage::allocate(q_bytes, device);
    q_storage->shape = {static_cast<dim_t>(packed_rows), static_cast<dim_t>(cols)};
    q_storage->dtype = DataType::Int8;
    q_storage->size_bytes = q_bytes;
    q_storage->capacity_bytes = q_bytes;
    q_storage->device = device;

    auto scale_storage = TensorStorage::allocate(scale_bytes, device);
    scale_storage->shape = {static_cast<dim_t>(num_blocks), static_cast<dim_t>(cols)};
    scale_storage->dtype = DataType::Float32;
    scale_storage->size_bytes = scale_bytes;
    scale_storage->capacity_bytes = scale_bytes;
    scale_storage->device = device;

    if (device == DeviceType::CPU) {
        std::memcpy(q_storage->data, packed_host.data(), q_bytes);
        std::memcpy(scale_storage->data, scale_host.data(), scale_bytes);
    } else {
        auto t = internal::get_memory_transfer(device);
        if (t.copy_h2d) {
            t.copy_h2d(q_storage->data, packed_host.data(), q_bytes);
            t.copy_h2d(scale_storage->data, scale_host.data(), scale_bytes);
        } else {
            throw std::runtime_error("quantize_storage_int4: no H2D transfer for device");
        }
    }

    return {std::move(q_storage), std::move(scale_storage)};
}

} // namespace velomind
