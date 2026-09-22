#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cmath>
#include <cstddef>

#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"
#include "internal/registry/shape.h"

namespace velomind::backend::cpu {

namespace {

template <typename T1, typename T2, typename T3>
void rmsnorm_impl(const TensorStorage* const* in,
                  TensorStorage* const*      out,
                  const void*                 attrs_ptr) {

    const auto& desc  = *static_cast<const OpDescriptor*>(attrs_ptr);
    const auto& attrs = std::get<RMSNormAttrs>(desc.attrs);

    const T1* x       = static_cast<const T1*>(in[0]->data);
    const T2* weight  = static_cast<const T2*>(in[1]->data);
    T3*       y       = static_cast<T3*>(out[0]->data);
    const auto& shape = in[0]->shape;

    std::size_t rank = shape.size();
    int axis = attrs.axis < 0 ? attrs.axis + static_cast<int>(rank)
                              : attrs.axis;

    if (axis != static_cast<int>(rank) - 1) {

        return;
    }

    std::size_t reduce_dim = static_cast<std::size_t>(shape[axis]);
    std::size_t outer      = storage_numel(*in[0]) / reduce_dim;

    for (std::size_t row = 0; row < outer; ++row) {
        const T1* x_row = x + row * reduce_dim;
        T3*       y_row = y + row * reduce_dim;

        float sumsq = 0.0f;
        for (std::size_t i = 0; i < reduce_dim; ++i) {
            const float v = static_cast<float>(x_row[i]);
            sumsq += v * v;
        }
        float mean_sq = sumsq / static_cast<float>(reduce_dim);
        float inv_rms = 1.0f / std::sqrt(mean_sq + static_cast<float>(attrs.epsilon));

        for (std::size_t i = 0; i < reduce_dim; ++i) {
            float val = static_cast<float>(x_row[i]) * inv_rms *
                        static_cast<float>(weight[i]);
            y_row[i] = static_cast<T3>(val);
        }
    }
}

VELOMIND_REGISTER_BINARY_KERNEL(DeviceType::CPU, Op::RMSNorm,
                                DataType::Float32, DataType::Float32,
                                DataType::Float32, rmsnorm_impl);

VELOMIND_REGISTER_BINARY_KERNEL(DeviceType::CPU, Op::RMSNorm,
                                DataType::Float16, DataType::Float16,
                                DataType::Float16, rmsnorm_impl);
VELOMIND_REGISTER_BINARY_KERNEL(DeviceType::CPU, Op::RMSNorm,
                                DataType::Float32, DataType::Float16,
                                DataType::Float32, rmsnorm_impl);

VELOMIND_REGISTER_BINARY_KERNEL(DeviceType::CPU, Op::RMSNorm,
                                DataType::BFloat16, DataType::BFloat16,
                                DataType::BFloat16, rmsnorm_impl);
VELOMIND_REGISTER_BINARY_KERNEL(DeviceType::CPU, Op::RMSNorm,
                                DataType::Float32, DataType::BFloat16,
                                DataType::Float32, rmsnorm_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::RMSNorm, ::velomind::detail::signature_rmsnorm)

}
