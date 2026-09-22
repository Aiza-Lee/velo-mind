#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../cuda_context.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"

namespace velomind::backend::cuda {

inline auto cuda_error_context(Op op, const TensorStorage& output, const char* stage)
    -> std::string {
    std::string message = std::string(op_name(op)) + " on CUDA output shape=[";
    for (std::size_t i = 0; i < output.shape.size(); ++i) {
        if (i) message += ',';
        message += std::to_string(output.shape[i]);
    }
    return message + "] at " + stage;
}

inline void check_cuda(cudaError_t result, Op op, const TensorStorage& output,
                       const char* stage) {
    const auto context = cuda_error_context(op, output, stage);
    if (result != cudaSuccess)
        throw std::runtime_error(context + ": " + cudaGetErrorName(result) +
                                 " (" + cudaGetErrorString(result) + ")");
    if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
        const auto key = std::string("CUDA:") + op_name(op) + ':' + stage;
        if (key == injected) throw std::runtime_error(context + ": injected failure");
    }
}

inline void check_cuda_kernel(Op op, const TensorStorage& output) {
    const auto launch = cudaGetLastError();
    check_cuda(launch, op, output, "launch");

    // 检查测试注入的同步阶段故障
    if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
        const auto key = std::string("CUDA:") + op_name(op) + ":sync";
        if (key == injected) {
            check_cuda(cudaSuccess, op, output, "sync");
        }
    }

    if (is_cuda_sync_debug_enabled()) {
        check_cuda(cudaStreamSynchronize(get_cuda_context().stream_handle()), op, output, "sync");
    }
}

}
