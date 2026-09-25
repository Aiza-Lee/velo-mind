#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void embedding_f32_i32_ispc(const float* table,
                                       const int*   indices,
                                       float*       y,
                                       int          hidden,
                                       int          total_tokens);

template <typename TTable, typename TIdx, typename TOut>
void embedding_ispc_impl(const TensorStorage* const* in,
                         TensorStorage* const*      out,
                         const void* ) {
    const auto* table = static_cast<const float*>(in[0]->data);
    const auto* idx   = static_cast<const int*>(in[1]->data);
    auto*       y     = static_cast<float*>(out[0]->data);

    const std::size_t hidden   = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t num_idx  = storage_numel(*in[1]);

    embedding_f32_i32_ispc(table, idx, y,
                           static_cast<int>(hidden),
                           static_cast<int>(num_idx));
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::ISPC, Op::Embedding,
                                DataType::Float32, DataType::Int32,
                                DataType::Float32, embedding_ispc_impl)

}

VELOMIND_REGISTER_SIGNATURE(Op::Embedding, ::velomind::detail::signature_embedding)

}
