#include "velomind/dtype.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "velomind/device.h"
#include "internal/registry/kernel.h"

namespace velomind {

auto is_dtype_supported(DeviceType device, DataType dtype) noexcept -> bool {
    if (!Device(device).is_available()) return false;
    switch (device) {
        case DeviceType::CPU:
            return dtype == DataType::Float32 || dtype == DataType::Int32 ||
                   dtype == DataType::Int8    || dtype == DataType::Bool  ||
                   dtype == DataType::Float16 || dtype == DataType::BFloat16;
        case DeviceType::ISPC:
            return dtype == DataType::Float32 || dtype == DataType::Int32 ||
                   dtype == DataType::Int8    || dtype == DataType::Bool  ||
                   dtype == DataType::Float16 || dtype == DataType::BFloat16;
        case DeviceType::CUDA:
            return dtype == DataType::Float32 || dtype == DataType::Int32 ||
                   dtype == DataType::Float16 || dtype == DataType::BFloat16;
        case DeviceType::VULKAN:
            return dtype == DataType::Float32 || dtype == DataType::Int32 ||
                   dtype == DataType::Int8    || dtype == DataType::Bool  ||
                   dtype == DataType::Float16 || dtype == DataType::BFloat16;
    }
    return false;
}

auto is_op_dtype_supported(
    DeviceType device, Op op,
    std::span<const DataType> input_dtypes,
    DataType output_dtype) noexcept -> bool
{
    if (!Device(device).is_available()) return false;
    auto fn = internal::resolve_op_kernel(op, input_dtypes, output_dtype, device);
    return fn != nullptr;
}

namespace {

template <typename TSrc, typename TDst>
void convert_typed_buffer(const void* src, void* dst, std::size_t numel) {
    const auto* s = static_cast<const TSrc*>(src);
    auto*       d = static_cast<TDst*>(dst);
    for (std::size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<TSrc, TDst>) {
            d[i] = s[i];
        } else if constexpr (std::is_same_v<TSrc, bool> || std::is_same_v<TDst, bool>) {
            d[i] = static_cast<TDst>(static_cast<bool>(s[i]));
        } else {
            d[i] = static_cast<TDst>(static_cast<float>(s[i]));
        }
    }
}

template <typename TSrc>
void dispatch_dst(const void* src, DataType dst_dtype, void* dst, std::size_t numel) {
    switch (dst_dtype) {
        case DataType::Float32:  convert_typed_buffer<TSrc, float>(src, dst, numel); break;
        case DataType::Float16:  convert_typed_buffer<TSrc, float16_t>(src, dst, numel); break;
        case DataType::BFloat16: convert_typed_buffer<TSrc, bfloat16_t>(src, dst, numel); break;
        case DataType::Int32:    convert_typed_buffer<TSrc, std::int32_t>(src, dst, numel); break;
        case DataType::Int8:     convert_typed_buffer<TSrc, std::int8_t>(src, dst, numel); break;
        case DataType::Bool:     convert_typed_buffer<TSrc, bool>(src, dst, numel); break;
    }
}

} // namespace

auto convert_dtype(
    const void* src, DataType src_dtype,
    void*       dst, DataType dst_dtype,
    std::size_t numel) -> void
{
    if (numel == 0) return;
    if (src == nullptr || dst == nullptr) {
        throw std::invalid_argument("convert_dtype: null buffer pointer");
    }
    if (src_dtype == dst_dtype) {
        std::memcpy(dst, src, numel * data_type_size(src_dtype));
        return;
    }
    switch (src_dtype) {
        case DataType::Float32:  dispatch_dst<float>(src, dst_dtype, dst, numel); break;
        case DataType::Float16:  dispatch_dst<float16_t>(src, dst_dtype, dst, numel); break;
        case DataType::BFloat16: dispatch_dst<bfloat16_t>(src, dst_dtype, dst, numel); break;
        case DataType::Int32:    dispatch_dst<std::int32_t>(src, dst_dtype, dst, numel); break;
        case DataType::Int8:     dispatch_dst<std::int8_t>(src, dst_dtype, dst, numel); break;
        case DataType::Bool:     dispatch_dst<bool>(src, dst_dtype, dst, numel); break;
    }
}

} // namespace velomind
