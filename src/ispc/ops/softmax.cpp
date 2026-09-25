#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <type_traits>

#include "internal/registry/kernel.h"
#include "internal/softmax.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void softmax_f32_ispc(const float* x,
                                 float*       y,
                                 int          cols,
                                 int          total_rows,
                                 int          inner);

template <typename T1, typename T2>
void softmax_ispc_impl(const TensorStorage* const* in,
                       TensorStorage* const*      out,
                       const void* attrs) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float>,
                  "ISPC Softmax currently supports Float32");

    const auto* x = static_cast<const float*>(in[0]->data);
    auto*       y = static_cast<float*>(out[0]->data);

    const auto layout = detail::softmax_layout(*in[0], attrs);
    const auto cols = layout.cols;
    const auto inner = layout.inner;
    const auto rows = layout.outer * inner;

    if (storage_numel(*in[0]) > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Softmax ISPC: tensor exceeds index range");
    softmax_f32_ispc(x, y, static_cast<int>(cols), static_cast<int>(rows), static_cast<int>(inner));
}

VELOMIND_REGISTER_UNARY_OP(DeviceType::ISPC, Op::Softmax,
                               DataType::Float32, DataType::Float32,
                               softmax_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Softmax, ::velomind::detail::signature_softmax)

}
