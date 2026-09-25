#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

// 沿指定轴以连续块复制方式扩展分组头，主要支撑 GQA 中 KV 头的广播对齐。
template <typename T1, typename T2>
void repeat_kv_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void*                attrs_ptr) {
    static_assert(std::is_same_v<T1, T2>, "RepeatKV: in/out dtype must match");
    const auto& desc = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attr = std::get<RepeatKVAttrs>(desc.attrs);

    const T1* x = static_cast<const T1*>(in[0]->data);
    T2*       y = static_cast<T2*>(out[0]->data);

    if (storage_numel(*out[0]) == 0) return;

    const auto& in_shape = in[0]->shape;
    const std::size_t rank = in_shape.size();
    const int axis = attr.axis < 0 ? attr.axis + static_cast<int>(rank) : attr.axis;
    const std::size_t repeats = static_cast<std::size_t>(attr.repeats);
    const std::size_t kv_heads = static_cast<std::size_t>(in_shape[axis]);

    std::size_t outer = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
        outer *= static_cast<std::size_t>(in_shape[i]);
    }
    std::size_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < rank; ++i) {
        inner *= static_cast<std::size_t>(in_shape[i]);
    }

    const std::size_t q_heads = kv_heads * repeats;
    const std::size_t inner_bytes = inner * sizeof(T1);

    for (std::size_t o = 0; o < outer; ++o) {
        const T1* in_outer  = x + o * kv_heads * inner;
        T2*       out_outer = y + o * q_heads * inner;
        for (std::size_t kv = 0; kv < kv_heads; ++kv) {
            const T1* src = in_outer + kv * inner;
            for (std::size_t r = 0; r < repeats; ++r) {
                T2* dst = out_outer + (kv * repeats + r) * inner;
                std::memcpy(dst, src, inner_bytes);
            }
        }
    }
}

VELOMIND_REGISTER_UNARY_SAME(DeviceType::CPU, Op::RepeatKV, repeat_kv_impl);

VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::RepeatKV,
                               DataType::Float16, DataType::Float16,
                               repeat_kv_impl);
VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::RepeatKV,
                               DataType::BFloat16, DataType::BFloat16,
                               repeat_kv_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::RepeatKV, ::velomind::detail::signature_repeat_kv)

}
