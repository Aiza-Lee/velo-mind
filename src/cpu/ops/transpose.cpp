#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

template <typename T1, typename T2>
void transpose_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void*                 attrs_ptr) {
    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& tattr = std::get<TransposeAttrs>(desc.attrs);

    const T1* x = static_cast<const T1*>(in[0]->data);
    T2*       y = static_cast<T2*>(out[0]->data);

    if (x == y || storage_numel(*out[0]) == 0) return;

    const auto& in_shape  = in[0]->shape;
    const auto& out_shape = out[0]->shape;
    const std::size_t rank = in_shape.size();

    std::vector<int> perm = tattr.perm;
    if (perm.empty()) {
        perm.resize(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            perm[i] = static_cast<int>(rank - 1 - i);
        }
    }

    const auto raw_in_strides = in[0]->effective_strides();
    std::vector<std::size_t> in_strides(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        in_strides[i] = static_cast<std::size_t>(raw_in_strides[i]);
    }
    const std::size_t total = storage_numel(*out[0]);
    std::vector<std::size_t> out_coords(rank, 0);
    std::vector<std::size_t> out_strides(rank, 1);
    for (std::size_t i = rank; i-- > 1;) {
        out_strides[i - 1] = out_strides[i] *
                            static_cast<std::size_t>(out_shape[i]);
    }

    for (std::size_t flat = 0; flat < total; ++flat) {
        std::size_t rem = flat;
        for (std::size_t ax = 0; ax < rank; ++ax) {
            out_coords[ax] = rem / out_strides[ax];
            rem          %= out_strides[ax];
        }

        std::size_t in_flat = 0;
        for (std::size_t ax = 0; ax < rank; ++ax) {
            const std::size_t in_axis = static_cast<std::size_t>(perm[ax]);
            in_flat += out_coords[ax] * in_strides[in_axis];
        }
        y[flat] = static_cast<T2>(x[in_flat]);
    }
}

VELOMIND_REGISTER_UNARY_SAME(DeviceType::CPU, Op::Transpose, transpose_impl);

VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::Transpose,
                               DataType::Float16, DataType::Float16,
                               transpose_impl);
VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::Transpose,
                               DataType::BFloat16, DataType::BFloat16,
                               transpose_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Transpose, ::velomind::detail::signature_transpose)

}
