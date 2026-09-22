#include "cuda_context.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace velomind::backend::cuda {

CudaStream::CudaStream(unsigned int flags) {
    cudaError_t err = cudaStreamCreateWithFlags(&_stream, flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("velomind: failed to create CUDA stream (") +
            cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
    }
}

CudaStream::~CudaStream() {
    if (_stream != nullptr) {
        cudaStreamDestroy(_stream);
        _stream = nullptr;
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : _stream(other._stream) {
    other._stream = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (_stream != nullptr) {
            cudaStreamDestroy(_stream);
        }
        _stream = other._stream;
        other._stream = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() const {
    if (_stream == nullptr) return;
    cudaError_t err = cudaStreamSynchronize(_stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("velomind: CUDA stream synchronize failed (") +
            cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
    }
}

CudaContext::CudaContext(unsigned int stream_flags) : _stream(stream_flags) {
    cublasStatus_t status = cublasCreate(&_cublas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("velomind: cublasCreate failed with status ") +
            std::to_string(static_cast<int>(status)));
    }
    status = cublasSetStream(_cublas, _stream.get());
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(_cublas);
        _cublas = nullptr;
        throw std::runtime_error(
            std::string("velomind: cublasSetStream failed with status ") +
            std::to_string(static_cast<int>(status)));
    }
}

CudaContext::~CudaContext() {
    if (_cublas != nullptr) {
        cublasDestroy(_cublas);
        _cublas = nullptr;
    }
}

CudaContext::CudaContext(CudaContext&& other) noexcept
    : _stream(std::move(other._stream)), _cublas(other._cublas) {
    other._cublas = nullptr;
}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        if (_cublas != nullptr) {
            cublasDestroy(_cublas);
        }
        _stream = std::move(other._stream);
        _cublas = other._cublas;
        other._cublas = nullptr;
    }
    return *this;
}

void CudaContext::synchronize() const {
    _stream.synchronize();
}

namespace {
thread_local CudaContext* t_active_context = nullptr;
}

ScopedCudaContext::ScopedCudaContext(CudaContext* ctx) : _prev_ctx(t_active_context) {
    t_active_context = ctx;
}

ScopedCudaContext::~ScopedCudaContext() {
    t_active_context = _prev_ctx;
}

auto get_cuda_context() -> CudaContext& {
    if (t_active_context != nullptr) {
        return *t_active_context;
    }
    static thread_local CudaContext t_default_context;
    return t_default_context;
}

auto is_cuda_sync_debug_enabled() -> bool {
    const char* env = std::getenv("VELOMIND_CUDA_SYNC");
    return env && (*env == '1' || *env == 't' || *env == 'T');
}

} // namespace velomind::backend::cuda
