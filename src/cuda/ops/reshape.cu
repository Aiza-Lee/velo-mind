#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cuda_runtime.h>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

template <typename T>
void reshape_impl(const TensorStorage* const* in,
                  TensorStorage* const*      out,
                  const void*                 ) {
    const auto* x = static_cast<const T*>(in[0]->data);
    auto*       y = static_cast<T*>(out[0]->data);

    std::size_t n = storage_numel(*out[0]);
    if (n == 0 || x == y) return;

    auto stream = get_cuda_context().stream_handle();
    check_cuda(cudaMemcpyAsync(y, x, n * sizeof(T), cudaMemcpyDeviceToDevice, stream),
               Op::Reshape, *out[0], "copy");

    if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
        const auto key = std::string("CUDA:Reshape:sync");
        if (key == injected) check_cuda(cudaSuccess, Op::Reshape, *out[0], "sync");
    }

    if (is_cuda_sync_debug_enabled()) {
        check_cuda(cudaStreamSynchronize(stream), Op::Reshape, *out[0], "sync");
    }
}

VELOMIND_REGISTER_UNARY_1T_FLOATS(DeviceType::CUDA, Op::Reshape, reshape_impl);
VELOMIND_REGISTER_UNARY_1T(DeviceType::CUDA, Op::Reshape, DataType::Int32, reshape_impl);

}

}
