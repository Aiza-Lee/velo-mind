#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "velomind/types.h"

namespace velomind::backend::cuda {

// RAII 管理 CUDA Stream 生命周期
class CudaStream {
public:
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking);
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    [[nodiscard]] auto get() const noexcept -> cudaStream_t { return _stream; }
    auto synchronize() const -> void;

private:
    cudaStream_t _stream = nullptr;
};

// RAII 管理 CUDA 执行上下文，统一持有流与绑定的 cuBLAS 句柄
class CudaContext {
public:
    explicit CudaContext(unsigned int stream_flags = cudaStreamNonBlocking);
    ~CudaContext();

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;

    [[nodiscard]] auto stream_handle() const noexcept -> cudaStream_t { return _stream.get(); }
    [[nodiscard]] auto cublas() const noexcept -> cublasHandle_t { return _cublas; }
    [[nodiscard]] auto stream() const noexcept -> const CudaStream& { return _stream; }

    auto synchronize() const -> void;

private:
    CudaStream _stream;
    cublasHandle_t _cublas = nullptr;
};

// 线程局部上下文作用域守卫，在当前作用域内覆盖活动上下文
class ScopedCudaContext {
public:
    explicit ScopedCudaContext(CudaContext* ctx);
    ~ScopedCudaContext();

    ScopedCudaContext(const ScopedCudaContext&) = delete;
    ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;

private:
    CudaContext* _prev_ctx = nullptr;
};

// 获取当前线程活动上下文；未显式设置时回退至线程局部默认上下文
auto get_cuda_context() -> CudaContext&;

// 检查环境变量 VELOMIND_CUDA_SYNC 是否启用了逐算子同步调试模式
auto is_cuda_sync_debug_enabled() -> bool;

} // namespace velomind::backend::cuda
