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

extern "C" void div_f32_ispc(float* a, float* b, float* c, int n);

template <typename T1, typename T2, typename T3>
void div_ispc_impl(const TensorStorage* const* in,
                   TensorStorage* const*      out,
                   const void*                 ) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float> && std::is_same_v<T3, float>,
                  "ISPC Div only supports float32");
    const float* a = static_cast<const float*>(in[0]->data);
    const float* b = static_cast<const float*>(in[1]->data);
    float*       c = static_cast<float*>(out[0]->data);
    int          n = static_cast<int>(storage_numel(*out[0]));

    div_f32_ispc(const_cast<float*>(a), const_cast<float*>(b), c, n);
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::ISPC, Op::Div,
                                DataType::Float32, DataType::Float32,
                                DataType::Float32, div_ispc_impl)

}

VELOMIND_REGISTER_SIGNATURE(Op::Div, ::velomind::detail::signature_elementwise)

}
