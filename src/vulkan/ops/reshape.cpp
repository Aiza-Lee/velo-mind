#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/shape_helpers.h"
#include "internal/vulkan_backend.h"

namespace velomind::backend::vulkan {

namespace {

    void reshape_impl(const TensorStorage* const* in,
                      TensorStorage* const*      out,
                      const void*                 ) {
        backend_vulkan::sync_vulkan_batch_if_pending();

        const float* x = static_cast<const float*>(in[0]->data);
        float*       y = static_cast<float*>(out[0]->data);
        const std::size_t n_in  = storage_numel(*in[0]);
        const std::size_t n_out = storage_numel(*out[0]);
        if (n_in != n_out) {
            throw std::runtime_error("velomind::Vulkan::Reshape: numel mismatch");
        }
        if (x == y || n_in == 0) return;
        std::memcpy(y, x, n_in * sizeof(float));
    }

    VELOMIND_REGISTER_UNARY_FN(DeviceType::VULKAN, Op::Reshape, DataType::Float32, DataType::Float32, &reshape_impl);

}

VELOMIND_REGISTER_SIGNATURE(Op::Reshape, ::velomind::detail::signature_reshape)

}
