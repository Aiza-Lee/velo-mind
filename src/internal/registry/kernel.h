#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/types.h"

namespace velomind::internal {

    // 输入类型数组固定为四项；num_inputs 指明实际参与匹配的前缀长度。
    struct KernelDtypeKey {
        std::array<DataType, 4> input_dtypes{};
        std::size_t             num_inputs   = 0;
        DataType                output_dtype = DataType::Float32;
    };

    auto operator==(const KernelDtypeKey& a, const KernelDtypeKey& b) noexcept -> bool;

    // 算子内核注册条目，封装类型匹配键与执行函数指针
    struct KernelEntry {
        KernelDtypeKey       key;
        Executable::KernelFn fn = nullptr;
    };

    // 获取指定设备的算子内核注册表（按 Op 枚举值索引）
    auto op_kernels(DeviceType device)
        -> std::array<std::vector<KernelEntry>, MAX_OPS>&;

    // 根据算子类型、输入参数类型前缀与输出类型在指定设备上检索匹配的内核执行函数
    auto resolve_op_kernel(Op                        op,
                        std::span<const DataType> input_dtypes,
                        DataType                  output_dtype,
                        DeviceType                device) -> Executable::KernelFn;

    // 向指定设备注册算子内核执行函数
    auto register_op_kernel(DeviceType           device,
                            Op                   op,
                            KernelDtypeKey       k,
                            Executable::KernelFn fn) -> void;


    // DataType 到 C++ 编译期原生标量类型的映射萃取模板
    template <DataType D> struct dtype_to_type;
    template <> struct dtype_to_type<DataType::Float32>  { using type = float; };
    template <> struct dtype_to_type<DataType::Int32>    { using type = std::int32_t; };
    template <> struct dtype_to_type<DataType::Int8>     { using type = std::int8_t; };
    template <> struct dtype_to_type<DataType::Bool>     { using type = bool; };
    template <> struct dtype_to_type<DataType::Float16>  { using type = float16_t; };
    template <> struct dtype_to_type<DataType::BFloat16> { using type = bfloat16_t; };

    template <DataType D>
    using dtype_t = typename dtype_to_type<D>::type;

} // namespace velomind::internal


// ============================================================================
// 算子内核便捷注册宏
// ============================================================================

// 宏标识符拼接辅助
#define VELOMIND_CONCAT_INNER_(a, b) a##b
#define VELOMIND_CONCAT_(a, b) VELOMIND_CONCAT_INNER_(a, b)

// ----------------------------------------------------------------------------
// 对外公开注册宏（现代极简命名）
// ----------------------------------------------------------------------------

// 底层通用算子注册宏：在静态作用域内原地执行 lambda 调用 register_op_kernel，
// 适用于多输入或非规则签名的算子内核注册。
#define VELOMIND_REGISTER_OP_KERNEL(DEVICE, OP_ENUM, KEY, ...)                            \
    [[maybe_unused]] static const auto VELOMIND_CONCAT_(_velomind_kr_, __LINE__) = [] {  \
        ::velomind::internal::register_op_kernel((DEVICE), (OP_ENUM), (KEY), (__VA_ARGS__)); \
        return true;                                                                     \
    }();

// 非模板普通函数指针快捷注册宏（如 Vulkan 管线调度函数）
#define VELOMIND_REGISTER_UNARY_FN(DEVICE, OP_ENUM, IN_DT, OUT_DT, FN)                   \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (IN_DT) },                                                                 \
            1,                                                                           \
            (OUT_DT)                                                                     \
        }),                                                                              \
        (FN))

#define VELOMIND_REGISTER_BINARY_FN(DEVICE, OP_ENUM, IN1_DT, IN2_DT, OUT_DT, FN)         \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (IN1_DT), (IN2_DT) },                                                      \
            2,                                                                           \
            (OUT_DT)                                                                     \
        }),                                                                              \
        (FN))

#define VELOMIND_REGISTER_TERNARY_FN(DEVICE, OP_ENUM, IN1_DT, IN2_DT, IN3_DT, OUT_DT, FN) \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (IN1_DT), (IN2_DT), (IN3_DT) },                                            \
            3,                                                                           \
            (OUT_DT)                                                                     \
        }),                                                                              \
        (FN))

// 单模板参数函数快捷注册宏（针对 template <typename T> 的同构内核）
#define VELOMIND_REGISTER_UNARY_1T(DEVICE, OP_ENUM, DT, NAME)                            \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (DT) },                                                                    \
            1,                                                                           \
            (DT)                                                                         \
        }),                                                                              \
        (&NAME<::velomind::internal::dtype_t<DT>>))

#define VELOMIND_REGISTER_BINARY_1T(DEVICE, OP_ENUM, DT, NAME)                           \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (DT), (DT) },                                                              \
            2,                                                                           \
            (DT)                                                                         \
        }),                                                                              \
        (&NAME<::velomind::internal::dtype_t<DT>>))

#define VELOMIND_REGISTER_TERNARY_SAME_1T(DEVICE, OP_ENUM, DT, NAME)                     \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (DT), (DT), (DT) },                                                        \
            3,                                                                           \
            (DT)                                                                         \
        }),                                                                              \
        (&NAME<::velomind::internal::dtype_t<DT>>))

// 多模板参数函数快捷注册宏（针对 template <typename ...> 的跨精度/泛型内核）
#define VELOMIND_REGISTER_UNARY_OP(DEVICE, OP_ENUM, IN_DT, OUT_DT, NAME)                 \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (IN_DT) },                                                                 \
            1,                                                                           \
            (OUT_DT)                                                                     \
        }),                                                                              \
        (&NAME<                                                                          \
            ::velomind::internal::dtype_t<IN_DT>,                                        \
            ::velomind::internal::dtype_t<OUT_DT>>))

#define VELOMIND_REGISTER_BINARY_OP(DEVICE, OP_ENUM, IN1_DT, IN2_DT, OUT_DT, NAME)       \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        (OP_ENUM),                                                                       \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (IN1_DT), (IN2_DT) },                                                      \
            2,                                                                           \
            (OUT_DT)                                                                     \
        }),                                                                              \
        (&NAME<                                                                          \
            ::velomind::internal::dtype_t<IN1_DT>,                                       \
            ::velomind::internal::dtype_t<IN2_DT>,                                       \
            ::velomind::internal::dtype_t<OUT_DT>>))

#define VELOMIND_REGISTER_TERNARY_OP(DEVICE, OP_ENUM, IN1_DT, IN2_DT, IN3_DT, OUT_DT, NAME) \
    VELOMIND_REGISTER_OP_KERNEL(                                                             \
        (DEVICE),                                                                            \
        (OP_ENUM),                                                                           \
        (::velomind::internal::KernelDtypeKey{                                               \
            { (IN1_DT), (IN2_DT), (IN3_DT) },                                                \
            3,                                                                               \
            (OUT_DT)                                                                         \
        }),                                                                                  \
        (&NAME<                                                                              \
            ::velomind::internal::dtype_t<IN1_DT>,                                           \
            ::velomind::internal::dtype_t<IN2_DT>,                                           \
            ::velomind::internal::dtype_t<IN3_DT>>))

// 专用算子快捷宏：QuantizedMatMul（第二输入固定为 Int8 权重）
#define VELOMIND_REGISTER_QUANTIZED_MATMUL(DEVICE, TA_DT, TSCALE_DT, TC_DT, NAME)         \
    VELOMIND_REGISTER_OP_KERNEL(                                                         \
        (DEVICE),                                                                        \
        Op::QuantizedMatMul,                                                             \
        (::velomind::internal::KernelDtypeKey{                                           \
            { (TA_DT), ::velomind::DataType::Int8, (TSCALE_DT) },                        \
            3,                                                                           \
            (TC_DT)                                                                      \
        }),                                                                              \
        (&NAME<                                                                          \
            ::velomind::internal::dtype_t<TA_DT>,                                        \
            ::velomind::internal::dtype_t<TSCALE_DT>,                                    \
            ::velomind::internal::dtype_t<TC_DT>>))

// 浮点三类型（Float32, Float16, BFloat16）批量注册宏
#define VELOMIND_REGISTER_UNARY_1T_FLOATS(DEVICE, OP_ENUM, NAME)                         \
    VELOMIND_DETAIL_FLOATS_APPLY_(                                                       \
        VELOMIND_DETAIL_UNARY_1T_EMIT_, DEVICE, OP_ENUM, NAME)                           \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_TERNARY_SAME_1T_FLOATS(DEVICE, OP_ENUM, NAME)                  \
    VELOMIND_DETAIL_FLOATS_APPLY_(                                                       \
        VELOMIND_DETAIL_TERNARY_SAME_1T_EMIT_, DEVICE, OP_ENUM, NAME)                    \
    static_assert(true, "force semicolon")

// 基础数值四类型（Float32, Int32, Int8, Bool）批量注册宏
#define VELOMIND_REGISTER_UNARY_SAME(DEVICE, OP_ENUM, NAME)                              \
    VELOMIND_DETAIL_NUMERIC_DTYPES_APPLY_(                                               \
        VELOMIND_DETAIL_UNARY_SAME_EMIT_, DEVICE, OP_ENUM, NAME)                         \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_BINARY_SAME(DEVICE, OP_ENUM, NAME)                             \
    VELOMIND_DETAIL_NUMERIC_DTYPES_APPLY_(                                               \
        VELOMIND_DETAIL_BINARY_SAME_EMIT_, DEVICE, OP_ENUM, NAME)                        \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_UNARY_ALL_PAIRS(DEVICE, OP_ENUM, NAME)                         \
    VELOMIND_DETAIL_NUMERIC_DTYPES_APPLY_(                                               \
        VELOMIND_DETAIL_UNARY_PAIR_OUTER_EMIT_, DEVICE, OP_ENUM, NAME)                   \
    static_assert(true, "force semicolon")


// ----------------------------------------------------------------------------
// 内部展开辅助宏（仅供宏内部实现，避免污染上层命名空间）
// ----------------------------------------------------------------------------

#define VELOMIND_DETAIL_FLOATS_APPLY_(F, ...)                                            \
    F(Float32, __VA_ARGS__)                                                              \
    F(Float16, __VA_ARGS__)                                                              \
    F(BFloat16, __VA_ARGS__)

#define VELOMIND_DETAIL_UNARY_1T_EMIT_(DT, DEVICE, OP_ENUM, NAME)                       \
    namespace VELOMIND_CONCAT_(_velomind_kr_u1t_, DT) {                                  \
        VELOMIND_REGISTER_UNARY_1T(                                                      \
            (DEVICE), (OP_ENUM), ::velomind::DataType::DT, NAME);                        \
    }

#define VELOMIND_DETAIL_TERNARY_SAME_1T_EMIT_(DT, DEVICE, OP_ENUM, NAME)                 \
    namespace VELOMIND_CONCAT_(_velomind_kr_t1t_, DT) {                                  \
        VELOMIND_REGISTER_TERNARY_SAME_1T(                                               \
            (DEVICE), (OP_ENUM), ::velomind::DataType::DT, NAME);                        \
    }

#define VELOMIND_DETAIL_NUMERIC_DTYPES_APPLY_(F, ...)                                    \
    F(Float32, __VA_ARGS__)                                                              \
    F(Int32, __VA_ARGS__)                                                                \
    F(Int8, __VA_ARGS__)                                                                 \
    F(Bool, __VA_ARGS__)

#define VELOMIND_DETAIL_UNARY_SAME_EMIT_(DT, DEVICE, OP_ENUM, NAME)                      \
    namespace VELOMIND_CONCAT_(_velomind_kr_unary_, DT) {                                \
        VELOMIND_REGISTER_UNARY_OP(                                                      \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::DT, ::velomind::DataType::DT,                          \
            NAME)                                                                        \
    }

#define VELOMIND_DETAIL_BINARY_SAME_EMIT_(DT, DEVICE, OP_ENUM, NAME)                     \
    namespace VELOMIND_CONCAT_(_velomind_kr_binary_, DT) {                               \
        VELOMIND_REGISTER_BINARY_OP(                                                     \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::DT, ::velomind::DataType::DT,                          \
            ::velomind::DataType::DT,                                                    \
            NAME)                                                                        \
    }

#define VELOMIND_DETAIL_UNARY_PAIR_OUTER_EMIT_(IN_DT, DEVICE, OP_ENUM, NAME)            \
    VELOMIND_DETAIL_NUMERIC_DTYPES_APPLY_(                                               \
        VELOMIND_DETAIL_UNARY_PAIR_INNER_EMIT_, IN_DT, DEVICE, OP_ENUM, NAME)

#define VELOMIND_DETAIL_UNARY_PAIR_INNER_EMIT_(OUT_DT, IN_DT, DEVICE, OP_ENUM, NAME)      \
    namespace VELOMIND_CONCAT_(_velomind_kr_unary_pair_,                                 \
                                VELOMIND_CONCAT_(IN_DT, OUT_DT)) {                        \
        VELOMIND_REGISTER_UNARY_OP(                                                      \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::IN_DT, ::velomind::DataType::OUT_DT,                   \
            NAME)                                                                        \
    }

