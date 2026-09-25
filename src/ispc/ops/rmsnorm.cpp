#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <type_traits>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void rmsnorm_f32_ispc(const float* x,
                                 const float* weight,
                                 float*       y,
                                 int          reduce_dim,
                                 float        eps,
                                 int          total_rows);

template <typename T1, typename T2, typename T3>
void rmsnorm_ispc_impl(const TensorStorage* const* in,
                       TensorStorage* const*      out,
                       const void*                attrs_ptr) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float> && std::is_same_v<T3, float>,
                  "ISPC RMSNorm currently supports Float32");

    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attrs = std::get<RMSNormAttrs>(desc.attrs);
    const float eps   = static_cast<float>(attrs.epsilon);

    const auto* x      = static_cast<const float*>(in[0]->data);
    const auto* weight = static_cast<const float*>(in[1]->data);
    auto*       y      = static_cast<float*>(out[0]->data);

    const auto& shape = in[0]->shape;
    const std::size_t rank = shape.size();
    const int axis = attrs.axis < 0 ? attrs.axis + static_cast<int>(rank) : attrs.axis;

    if (axis != static_cast<int>(rank) - 1) {
        return;
    }

    const std::size_t reduce_dim = static_cast<std::size_t>(shape[axis]);
    const std::size_t total_rows = storage_numel(*in[0]) / reduce_dim;

    rmsnorm_f32_ispc(x, weight, y,
                     static_cast<int>(reduce_dim),
                     eps,
                     static_cast<int>(total_rows));
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::ISPC, Op::RMSNorm,
                                DataType::Float32, DataType::Float32,
                                DataType::Float32, rmsnorm_ispc_impl)

}

VELOMIND_REGISTER_SIGNATURE(Op::RMSNorm, ::velomind::detail::signature_rmsnorm)

}
