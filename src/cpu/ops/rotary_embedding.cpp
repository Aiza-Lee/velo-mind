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

template <typename TIn, typename TCos, typename TSin>
void rotary_embedding_impl(const TensorStorage* const* in,
                           TensorStorage* const*      out,
                           const void*                 ) {
    const TIn*  x   = static_cast<const TIn*>(in[0]->data);
    const TCos* cos = static_cast<const TCos*>(in[1]->data);
    const TSin* sin = static_cast<const TSin*>(in[2]->data);
    TIn*        y   = static_cast<TIn*>(out[0]->data);

    const std::size_t head_dim  = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t half_dim  = head_dim / 2;
    const std::size_t outer     = storage_numel(*in[0]) / head_dim;
    const std::size_t cos_rows  = (in[1]->shape.size() >= 2) ? static_cast<std::size_t>(in[1]->shape[0]) : 1;

    for (std::size_t i = 0; i < outer; ++i) {
        const TIn*  x_row   = x + i * head_dim;
        TIn*        y_row   = y + i * head_dim;
        const std::size_t pos = (cos_rows > 1) ? (i % cos_rows) : 0;
        const TCos* cos_row = cos + pos * half_dim;
        const TSin* sin_row = sin + pos * half_dim;
        for (std::size_t j = 0; j < half_dim; ++j) {
            const TIn a = x_row[j];
            const TIn b = x_row[j + half_dim];
            const TIn c = cos_row[j];
            const TIn s = sin_row[j];
            y_row[j]             = a * c - b * s;
            y_row[j + half_dim]  = b * c + a * s;
        }
    }
}

VELOMIND_REGISTER_TERNARY_OP(
    ::velomind::DeviceType::CPU,
    ::velomind::Op::RotaryEmbedding,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    rotary_embedding_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::RotaryEmbedding,
                             ::velomind::detail::signature_rotary_embedding)

}
