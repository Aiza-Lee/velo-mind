#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <type_traits>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void neg_f32_ispc(float* x, float* y, int n);

template <typename T1, typename T2>
void neg_ispc_impl(const TensorStorage* const* in,
                   TensorStorage* const*      out,
                   const void*                 ) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float>,
                  "ISPC Neg only supports float32");
    const float* x = static_cast<const float*>(in[0]->data);
    float*       y = static_cast<float*>(out[0]->data);
    int          n = static_cast<int>(storage_numel(*out[0]));

    neg_f32_ispc(const_cast<float*>(x), y, n);
}

VELOMIND_REGISTER_UNARY_KERNEL(DeviceType::ISPC, Op::Neg,
                               DataType::Float32, DataType::Float32,
                               neg_ispc_impl)

}

VELOMIND_REGISTER_SIGNATURE(Op::Neg, ::velomind::detail::signature_elementwise)

}
