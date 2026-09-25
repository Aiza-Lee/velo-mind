#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cmath>
#include <cstddef>

#include "internal/registry/kernel.h"
#include "internal/softmax.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

template <typename T1, typename T2>
void softmax_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* attrs) {
    const T1* x = static_cast<const T1*>(in[0]->data);
    T2*       y = static_cast<T2*>(out[0]->data);

    const auto layout = detail::softmax_layout(*in[0], attrs);
    const auto cols = layout.cols;
    const auto inner = layout.inner;
    const auto rows = layout.outer * inner;

    for (std::size_t r = 0; r < rows; ++r) {
        const T1* x_row = x + (r / inner) * cols * inner + r % inner;
        T2*       y_row = y + (r / inner) * cols * inner + r % inner;

        float row_max = static_cast<float>(x_row[0]);
        for (std::size_t j = 1; j < cols; ++j) {
            float v = static_cast<float>(x_row[j * inner]);
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            row_sum += std::exp(static_cast<float>(x_row[j * inner]) - row_max);
        }

        for (std::size_t j = 0; j < cols; ++j) {
            float val = std::exp(static_cast<float>(x_row[j * inner]) - row_max) / row_sum;
            y_row[j * inner] = static_cast<T2>(val);
        }
    }
}

VELOMIND_REGISTER_UNARY_SAME(DeviceType::CPU, Op::Softmax, softmax_impl);

VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::Softmax,
                               DataType::Float16, DataType::Float16,
                               softmax_impl);
VELOMIND_REGISTER_UNARY_OP(DeviceType::CPU, Op::Softmax,
                               DataType::BFloat16, DataType::BFloat16,
                               softmax_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Softmax, ::velomind::detail::signature_softmax)

}
