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

    struct KernelEntry {
        KernelDtypeKey       key;
        Executable::KernelFn fn = nullptr;
    };

    auto op_kernels(DeviceType device)
        -> std::array<std::vector<KernelEntry>, MAX_OPS>&;

    auto resolve_op_kernel(Op                        op,
                        std::span<const DataType> input_dtypes,
                        DataType                  output_dtype,
                        DeviceType                device) -> Executable::KernelFn;

    auto register_op_kernel(DeviceType           device,
                            Op                   op,
                            KernelDtypeKey       k,
                            Executable::KernelFn fn) -> void;

    struct KernelRegistrar {
        KernelRegistrar(DeviceType           device,
                        Op                   op,
                        KernelDtypeKey       k,
                        Executable::KernelFn fn);
    };

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


#define VELOMIND_CONCAT_INNER_(a, b) a##b
#define VELOMIND_CONCAT_(a, b) VELOMIND_CONCAT_INNER_(a, b)

#define VELOMIND_NUMERIC_DTYPES_X(F) \
    F(Float32)                       \
    F(Int32)                         \
    F(Int8)                          \
    F(Bool)

#define VELOMIND_REGISTER_KERNEL(DEVICE, OP_ENUM, KEY, FN)                               \
    [[maybe_unused]] static const auto VELOMIND_CONCAT_(_velomind_kr_, __LINE__) = [] {  \
        ::velomind::internal::register_op_kernel((DEVICE), (OP_ENUM), (KEY), (FN));      \
        return true;                                                                     \
    }();

#define VELOMIND_REGISTER_BINARY_KERNEL(DEVICE, OP_ENUM, IN1_DT, IN2_DT, OUT_DT, NAME)   \
    VELOMIND_REGISTER_KERNEL(                                                            \
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

#define VELOMIND_REGISTER_UNARY_KERNEL(DEVICE, OP_ENUM, IN_DT, OUT_DT, NAME)             \
    VELOMIND_REGISTER_KERNEL(                                                            \
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

#define VELOMIND_NUMERIC_DTYPES_APPLY_(F, ...)                                           \
    F(Float32, __VA_ARGS__)                                                              \
    F(Int32, __VA_ARGS__)                                                                \
    F(Int8, __VA_ARGS__)                                                                 \
    F(Bool, __VA_ARGS__)

#define VELOMIND_REGISTER_UNARY_KERNEL_SAME(DEVICE, OP_ENUM, NAME)                       \
    VELOMIND_NUMERIC_DTYPES_APPLY_(                                                      \
        VELOMIND_REGISTER_UNARY_SAME_EMIT_, DEVICE, OP_ENUM, NAME)                       \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_UNARY_SAME_EMIT_(DT, DEVICE, OP_ENUM, NAME)                    \
    namespace VELOMIND_CONCAT_(_velomind_kr_unary_, DT) {                                \
        VELOMIND_REGISTER_UNARY_KERNEL(                                                  \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::DT, ::velomind::DataType::DT,                          \
            NAME)                                                                        \
    }

#define VELOMIND_REGISTER_BINARY_KERNEL_SAME(DEVICE, OP_ENUM, NAME)                      \
    VELOMIND_NUMERIC_DTYPES_APPLY_(                                                      \
        VELOMIND_REGISTER_BINARY_SAME_EMIT_, DEVICE, OP_ENUM, NAME)                      \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_BINARY_SAME_EMIT_(DT, DEVICE, OP_ENUM, NAME)                   \
    namespace VELOMIND_CONCAT_(_velomind_kr_binary_, DT) {                               \
        VELOMIND_REGISTER_BINARY_KERNEL(                                                 \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::DT, ::velomind::DataType::DT,                          \
            ::velomind::DataType::DT,                                                    \
            NAME)                                                                        \
    }

#define VELOMIND_REGISTER_UNARY_KERNEL_ALL_PAIRS(DEVICE, OP_ENUM, NAME)                  \
    VELOMIND_NUMERIC_DTYPES_APPLY_(                                                      \
        VELOMIND_REGISTER_UNARY_PAIR_OUTER_EMIT_, DEVICE, OP_ENUM, NAME)                 \
    static_assert(true, "force semicolon")

#define VELOMIND_REGISTER_UNARY_PAIR_OUTER_EMIT_(IN_DT, DEVICE, OP_ENUM, NAME)           \
    VELOMIND_NUMERIC_DTYPES_APPLY_(                                                      \
        VELOMIND_REGISTER_UNARY_PAIR_INNER_EMIT_, IN_DT, DEVICE, OP_ENUM, NAME)

#define VELOMIND_REGISTER_UNARY_PAIR_INNER_EMIT_(OUT_DT, IN_DT, DEVICE, OP_ENUM, NAME)   \
    namespace VELOMIND_CONCAT_(_velomind_kr_unary_pair_,                                 \
                               VELOMIND_CONCAT_(IN_DT, OUT_DT)) {                        \
        VELOMIND_REGISTER_UNARY_KERNEL(                                                  \
            (DEVICE), (OP_ENUM),                                                         \
            ::velomind::DataType::IN_DT, ::velomind::DataType::OUT_DT,                   \
            NAME)                                                                        \
    }
