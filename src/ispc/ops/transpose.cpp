#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <vector>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void transpose_f32_ispc(const float* x,
                                   float*       y,
                                   const int*   out_strides,
                                   const int*   in_strides,
                                   const int*   inv_perm,
                                   int          rank,
                                   int          total);

template <typename T1, typename T2>
void transpose_ispc_impl(const TensorStorage* const* in,
                         TensorStorage* const*      out,
                         const void*                attrs_ptr) {
    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& tattr = std::get<TransposeAttrs>(desc.attrs);

    const auto* x = static_cast<const float*>(in[0]->data);
    auto*       y = static_cast<float*>(out[0]->data);

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

    std::vector<int> in_strides(rank, 1);
    for (std::size_t i = rank; i-- > 1;) {
        in_strides[i - 1] = in_strides[i] * static_cast<int>(in_shape[i]);
    }
    std::vector<int> out_strides(rank, 1);
    for (std::size_t i = rank; i-- > 1;) {
        out_strides[i - 1] = out_strides[i] * static_cast<int>(out_shape[i]);
    }

    const std::size_t total = storage_numel(*out[0]);

    transpose_f32_ispc(x, y,
                       out_strides.data(),
                       in_strides.data(),
                       perm.data(),
                       static_cast<int>(rank),
                       static_cast<int>(total));
}

VELOMIND_REGISTER_UNARY_KERNEL(DeviceType::ISPC, Op::Transpose,
                               DataType::Float32, DataType::Float32,
                               transpose_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Transpose, ::velomind::detail::signature_transpose)

}
