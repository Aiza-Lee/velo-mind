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

extern "C" void rotary_embedding_f32_ispc(const float* x,
                                          const float* cos_cache,
                                          const float* sin_cache,
                                          float*       y,
                                          int          head_dim,
                                          int          cos_rows,
                                          int          total_rows);

template <typename TIn, typename TCos, typename TSin>
void rotary_embedding_ispc_impl(const TensorStorage* const* in,
                                TensorStorage* const*      out,
                                const void* ) {
    const auto* x   = static_cast<const float*>(in[0]->data);
    const auto* cos = static_cast<const float*>(in[1]->data);
    const auto* sin = static_cast<const float*>(in[2]->data);
    auto*       y   = static_cast<float*>(out[0]->data);

    const std::size_t head_dim = static_cast<std::size_t>(in[0]->shape.back());
    const std::size_t outer    = storage_numel(*in[0]) / head_dim;
    const std::size_t cos_rows = (in[1]->shape.size() >= 2) ? static_cast<std::size_t>(in[1]->shape[0]) : 1;

    rotary_embedding_f32_ispc(x, cos, sin, y,
                              static_cast<int>(head_dim),
                              static_cast<int>(cos_rows),
                              static_cast<int>(outer));
}

VELOMIND_REGISTER_TERNARY_OP(
    ::velomind::DeviceType::ISPC,
    ::velomind::Op::RotaryEmbedding,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    ::velomind::DataType::Float32,
    rotary_embedding_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::RotaryEmbedding,
                            ::velomind::detail::signature_rotary_embedding)

}
