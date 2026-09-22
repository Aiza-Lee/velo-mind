#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "velomind/types.h"
#include "velomind_core_export.h"

#if defined(__CUDACC__)
#define VELOMIND_HD __host__ __device__
#else
#define VELOMIND_HD
#endif

namespace velomind {

// 标量级单精度与半精度浮点相互转换
VELOMIND_HD inline auto float16_to_float(std::uint16_t h) noexcept -> float {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1F;
    const std::uint32_t mant = h & 0x03FF;

    std::uint32_t f32 = 0;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign;
        } else {
            // 非规格化数：mant * 2^-24
            std::uint32_t m = mant;
            std::int32_t e = -14;
            while ((m & 0x0400) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x03FF;
            f32 = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        // 无穷大或 NaN
        f32 = sign | 0x7F800000 | (mant << 13);
        if (mant != 0) f32 |= 0x00400000;
    } else {
        // 规格化数
        f32 = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    union { std::uint32_t u; float f; } pun{f32};
    return pun.f;
}

VELOMIND_HD inline auto float_to_float16(float f) noexcept -> std::uint16_t {
    union { float f; std::uint32_t u; } pun{f};
    std::uint32_t x = pun.u;

    const std::uint32_t sign = (x >> 16) & 0x8000;
    const std::uint32_t raw_exp = (x >> 23) & 0xFF;
    const std::uint32_t mant = x & 0x007FFFFF;

    if (raw_exp == 0xFF) {
        if (mant != 0) return static_cast<std::uint16_t>(sign | 0x7E00);
        return static_cast<std::uint16_t>(sign | 0x7C00);
    }

    std::int32_t exp = static_cast<std::int32_t>(raw_exp) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        std::uint32_t m = (mant | 0x00800000) >> (1 - exp);
        if ((m & 0x1000) && ((m & 0x0FFF) || (m & 0x2000))) {
            m += 0x2000;
        }
        return static_cast<std::uint16_t>(sign | (m >> 13));
    } else if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00);
    }

    std::uint32_t m = mant;
    if ((m & 0x1000) && ((m & 0x0FFF) || (m & 0x2000))) {
        m += 0x2000;
        if (m & 0x00800000) {
            m = 0;
            exp++;
            if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (m >> 13));
}

VELOMIND_HD inline auto bfloat16_to_float(std::uint16_t b) noexcept -> float {
    std::uint32_t f32 = static_cast<std::uint32_t>(b) << 16;
    union { std::uint32_t u; float f; } pun{f32};
    return pun.f;
}

VELOMIND_HD inline auto float_to_bfloat16(float f) noexcept -> std::uint16_t {
    union { float f; std::uint32_t u; } pun{f};
    std::uint32_t x = pun.u;
    if ((x & 0x7F800000) == 0x7F800000 && (x & 0x007FFFFF) != 0) {
        return static_cast<std::uint16_t>((x >> 16) | 0x0040);
    }
    std::uint32_t lsb = (x >> 16) & 1;
    std::uint32_t bias = 0x7FFF + lsb;
    x += bias;
    return static_cast<std::uint16_t>(x >> 16);
}

// IEEE 754 半精度浮点数封装（1 位符号，5 位指数，10 位尾数）
struct float16_t {
    std::uint16_t bits = 0;

    constexpr float16_t() noexcept = default;
    VELOMIND_HD constexpr explicit float16_t(std::uint16_t raw_bits, bool) noexcept : bits(raw_bits) {}
    VELOMIND_HD float16_t(float f) noexcept : bits(float_to_float16(f)) {}

    static VELOMIND_HD constexpr auto from_bits(std::uint16_t b) noexcept -> float16_t {
        return float16_t(b, true);
    }
    VELOMIND_HD constexpr auto to_bits() const noexcept -> std::uint16_t { return bits; }

    VELOMIND_HD operator float() const noexcept { return float16_to_float(bits); }

    friend VELOMIND_HD constexpr auto operator==(float16_t a, float16_t b) noexcept -> bool { return a.bits == b.bits; }
    friend VELOMIND_HD constexpr auto operator!=(float16_t a, float16_t b) noexcept -> bool { return a.bits != b.bits; }
};

// Google Brain 16 位浮点数封装（1 位符号，8 位指数，7 位尾数）
struct bfloat16_t {
    std::uint16_t bits = 0;

    constexpr bfloat16_t() noexcept = default;
    VELOMIND_HD constexpr explicit bfloat16_t(std::uint16_t raw_bits, bool) noexcept : bits(raw_bits) {}
    VELOMIND_HD bfloat16_t(float f) noexcept : bits(float_to_bfloat16(f)) {}

    static VELOMIND_HD constexpr auto from_bits(std::uint16_t b) noexcept -> bfloat16_t {
        return bfloat16_t(b, true);
    }
    VELOMIND_HD constexpr auto to_bits() const noexcept -> std::uint16_t { return bits; }

    VELOMIND_HD operator float() const noexcept { return bfloat16_to_float(bits); }

    friend VELOMIND_HD constexpr auto operator==(bfloat16_t a, bfloat16_t b) noexcept -> bool { return a.bits == b.bits; }
    friend VELOMIND_HD constexpr auto operator!=(bfloat16_t a, bfloat16_t b) noexcept -> bool { return a.bits != b.bits; }
};

// 前向声明算子枚举
enum class Op : std::uint16_t;

// 查询目标硬件设备对特定数据类型的存储与计算支持状态
VELOMIND_CORE_EXPORT auto is_dtype_supported(DeviceType device, DataType dtype) noexcept -> bool;

// 查询特定硬件设备上特定算子输入输出数据类型组合是否已注册内核支持
VELOMIND_CORE_EXPORT auto is_op_dtype_supported(
    DeviceType device, Op op,
    std::span<const DataType> input_dtypes,
    DataType output_dtype) noexcept -> bool;

// 在不同数据类型缓冲区之间批量转换连续元素
VELOMIND_CORE_EXPORT auto convert_dtype(
    const void* src, DataType src_dtype,
    void* dst, DataType dst_dtype,
    std::size_t numel) -> void;

} // namespace velomind
