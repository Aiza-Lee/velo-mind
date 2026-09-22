#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

template <typename T1, typename T2, typename T3>
void div_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T1* a = static_cast<const T1*>(in[0]->data);
    const T2* b = static_cast<const T2*>(in[1]->data);
    T3*       c = static_cast<T3*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    for (std::size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T3>(a[i]) / static_cast<T3>(b[i]);
    }
}

VELOMIND_REGISTER_BINARY_KERNEL_SAME(DeviceType::CPU, Op::Div, div_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Div, ::velomind::detail::signature_elementwise)

}
