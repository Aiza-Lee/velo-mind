#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

template <typename T1, typename T2>
void reshape_impl(const TensorStorage* const* in,
                  TensorStorage* const*      out,
                  const void*                 ) {
    static_assert(std::is_same_v<T1, T2>,
                  "Reshape in/out dtypes must match (v1 uniform-precision)");
    const T1* x = static_cast<const T1*>(in[0]->data);
    T2*       y = static_cast<T2*>(out[0]->data);
    std::size_t n_in  = storage_numel(*in[0]);
    std::size_t n_out = storage_numel(*out[0]);
    if (n_in != n_out) {
        throw std::runtime_error("velomind::Reshape: numel mismatch");
    }
    if (x == y || n_in == 0) return;
    std::memcpy(y, x, n_in * sizeof(T2));
}

VELOMIND_REGISTER_UNARY_KERNEL_SAME(DeviceType::CPU, Op::Reshape, reshape_impl);

VELOMIND_REGISTER_UNARY_KERNEL(DeviceType::CPU, Op::Reshape,
                               DataType::Float16, DataType::Float16,
                               reshape_impl);
VELOMIND_REGISTER_UNARY_KERNEL(DeviceType::CPU, Op::Reshape,
                               DataType::BFloat16, DataType::BFloat16,
                               reshape_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Reshape, ::velomind::detail::signature_reshape)

}
