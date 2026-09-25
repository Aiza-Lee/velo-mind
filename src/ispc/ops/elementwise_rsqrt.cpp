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

extern "C" void rsqrt_f32_ispc(const float* x, float* y, int n);

template <typename T1, typename T2>
void rsqrt_ispc_impl(const TensorStorage* const* in,
                     TensorStorage* const*      out,
                     const void* ) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float>,
                  "ISPC Rsqrt currently supports Float32");

    const auto* x = static_cast<const float*>(in[0]->data);
    auto*       y = static_cast<float*>(out[0]->data);
    const std::size_t n = storage_numel(*out[0]);

    rsqrt_f32_ispc(x, y, static_cast<int>(n));
}

VELOMIND_REGISTER_UNARY_OP(DeviceType::ISPC, Op::Rsqrt,
                               DataType::Float32, DataType::Float32,
                               rsqrt_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Rsqrt, ::velomind::detail::signature_elementwise)

}
