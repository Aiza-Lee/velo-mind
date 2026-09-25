#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::ispc {

namespace {

extern "C" void matmul_f32_ispc(const float* a,
                                const float* b,
                                float*       c,
                                int          batch,
                                int          m,
                                int          k,
                                int          n);

template <typename T1, typename T2, typename T3>
void matmul_ispc_impl(const TensorStorage* const* in,
                      TensorStorage* const*      out,
                      const void* ) {
    static_assert(std::is_same_v<T1, float> && std::is_same_v<T2, float> && std::is_same_v<T3, float>,
                  "ISPC MatMul currently supports Float32");

    const auto* a = static_cast<const float*>(in[0]->data);
    const auto* b = static_cast<const float*>(in[1]->data);
    auto*       c = static_cast<float*>(out[0]->data);

    const auto& a_shape = in[0]->shape;
    const auto& b_shape = in[1]->shape;
    const auto& c_shape = out[0]->shape;

    const std::size_t a_rank = a_shape.size();
    const std::size_t b_rank = b_shape.size();
    const std::size_t c_rank = c_shape.size();

    if (a_rank < 2 || b_rank < 2 || c_rank < 2) {
        throw std::runtime_error("velomind::ispc::MatMul: rank must be >= 2");
    }

    const std::size_t m = static_cast<std::size_t>(a_shape[a_rank - 2]);
    const std::size_t k = static_cast<std::size_t>(a_shape[a_rank - 1]);
    const std::size_t n = static_cast<std::size_t>(b_shape[b_rank - 1]);

    if (static_cast<std::size_t>(b_shape[b_rank - 2]) != k) {
        throw std::runtime_error("velomind::ispc::MatMul: K dimension mismatch");
    }
    if (static_cast<std::size_t>(c_shape[c_rank - 2]) != m ||
        static_cast<std::size_t>(c_shape[c_rank - 1]) != n) {
        throw std::runtime_error("velomind::ispc::MatMul: output shape mismatch");
    }

    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < c_rank; ++i) {
        batch *= static_cast<std::size_t>(c_shape[i]);
    }

    matmul_f32_ispc(a, b, c,
                    static_cast<int>(batch),
                    static_cast<int>(m),
                    static_cast<int>(k),
                    static_cast<int>(n));
}

VELOMIND_REGISTER_BINARY_OP(DeviceType::ISPC, Op::MatMul,
                                DataType::Float32, DataType::Float32,
                                DataType::Float32, matmul_ispc_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::MatMul, ::velomind::detail::signature_matmul)

}
