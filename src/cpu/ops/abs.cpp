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

template <typename T1, typename T2>
void abs_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T1* x = static_cast<const T1*>(in[0]->data);
    T2*       y = static_cast<T2*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    for (std::size_t i = 0; i < n; ++i) {
        T2 v = static_cast<T2>(x[i]);
        y[i] = v < T2(0) ? static_cast<T2>(-v) : v;
    }
}

VELOMIND_REGISTER_UNARY_SAME(DeviceType::CPU, Op::Abs, abs_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Abs, ::velomind::detail::signature_elementwise)

}
