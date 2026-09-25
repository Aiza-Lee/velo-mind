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

template <typename TTable, typename TIdx, typename TOut>
void embedding_impl(const TensorStorage* const* in,
                    TensorStorage* const*      out,
                    const void* ) {
    const TTable* table = static_cast<const TTable*>(in[0]->data);
    const TIdx*   idx   = static_cast<const TIdx*>(in[1]->data);
    TOut*         y     = static_cast<TOut*>(out[0]->data);

    const std::size_t hidden  = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t num_idx = storage_numel(*in[1]);

    for (std::size_t i = 0; i < num_idx; ++i) {
        const std::size_t token = static_cast<std::size_t>(idx[i]);
        for (std::size_t j = 0; j < hidden; ++j) {
            y[i * hidden + j] = static_cast<TOut>(table[token * hidden + j]);
        }
    }
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::CPU, Op::Embedding,
                                DataType::Float32, DataType::Int32,
                                DataType::Float32, embedding_impl)

}

VELOMIND_REGISTER_SIGNATURE(Op::Embedding, ::velomind::detail::signature_embedding)

}
