#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::ispc {

namespace {

template <typename T1, typename T2, typename T3>
void concat_ispc_impl(const TensorStorage* const* in,
                      TensorStorage* const*      out,
                      const void*                 attrs_ptr) {
    static_assert(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>,
                  "Concat: in1/in2/out dtype must match (v1 uniform-precision)");
    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& cattr = std::get<ConcatAttrs>(desc.attrs);

    const T1* a = static_cast<const T1*>(in[0]->data);
    const T2* b = static_cast<const T2*>(in[1]->data);
    T3*       y = static_cast<T3*>(out[0]->data);

    const auto& a_shape = in[0]->shape;
    const auto& b_shape = in[1]->shape;
    const auto& o_shape = out[0]->shape;
    const std::size_t rank = a_shape.size();

    int axis = cattr.axis < 0 ? cattr.axis + static_cast<int>(rank)
                              : cattr.axis;

    for (std::size_t i = 0; i < rank; ++i) {
        if (i == static_cast<std::size_t>(axis)) continue;
        if (a_shape[i] != b_shape[i] || a_shape[i] != o_shape[i]) {
            throw std::runtime_error("velomind::Concat: non-axis dim mismatch");
        }
    }
    if (a_shape[axis] + b_shape[axis] != o_shape[axis]) {
        throw std::runtime_error("velomind::Concat: axis sum mismatch");
    }

    std::size_t outer = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
        outer *= static_cast<std::size_t>(a_shape[i]);
    }
    const std::size_t a_axis_len = static_cast<std::size_t>(a_shape[axis]);
    const std::size_t b_axis_len = static_cast<std::size_t>(b_shape[axis]);
    const std::size_t out_axis_len = a_axis_len + b_axis_len;
    std::size_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < rank; ++i) {
        inner *= static_cast<std::size_t>(a_shape[i]);
    }

    const std::size_t a_chunk_bytes = a_axis_len * inner * sizeof(T1);
    const std::size_t b_chunk_bytes = b_axis_len * inner * sizeof(T2);

    for (std::size_t o = 0; o < outer; ++o) {
        T3* y_chunk = y + o * out_axis_len * inner;
        const T1* a_chunk = a + o * a_axis_len * inner;
        const T2* b_chunk = b + o * b_axis_len * inner;
        std::memcpy(y_chunk, a_chunk, a_chunk_bytes);
        std::memcpy(y_chunk + a_axis_len * inner, b_chunk, b_chunk_bytes);
    }
}

VELOMIND_REGISTER_BINARY_SAME(DeviceType::ISPC, Op::Concat, concat_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Concat, ::velomind::detail::signature_concat)

}
