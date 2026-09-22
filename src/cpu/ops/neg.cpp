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
void neg_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
    const T1* a = static_cast<const T1*>(in[0]->data);
    T2*       c = static_cast<T2*>(out[0]->data);
    auto      n = storage_numel(*out[0]);

    for (std::size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T2>(-static_cast<T2>(a[i]));
    }
}

VELOMIND_REGISTER_UNARY_KERNEL_SAME(DeviceType::CPU, Op::Neg, neg_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Neg, ::velomind::detail::signature_elementwise)

}
