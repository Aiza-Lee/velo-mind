#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "cuda_context.h"
#include "internal/registry/memory_transfer.h"
#include "internal/registry/storage.h"

namespace velomind::backend::cuda {

namespace {

    auto cuda_allocator(std::size_t bytes) -> std::shared_ptr<TensorStorage> {
        void* dev = nullptr;
        if (bytes > 0) {
            cudaError_t err = cudaMalloc(&dev, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("velomind: cudaMalloc failed for ") +
                    std::to_string(bytes) + " bytes (" +
                    cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
            }
        }
        // 删除器捕获原始设备指针；bind_input 改写 data 后仍应释放原分配。
        auto s = std::shared_ptr<TensorStorage>(
            new TensorStorage{},
            [dev](TensorStorage* p) {
                if (dev != nullptr) cudaFree(dev);
                delete p;
            });
        s->size_bytes     = bytes;
        s->capacity_bytes = bytes;
        s->offset_bytes   = 0;
        s->device         = DeviceType::CUDA;
        s->data           = dev;
        return s;
    }

    void cuda_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
        if (bytes == 0) return;
        auto stream = get_cuda_context().stream_handle();
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("velomind: cudaMemcpyAsync (H2D) failed for ") +
                std::to_string(bytes) + " bytes (" +
                cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("velomind: cudaStreamSynchronize (H2D) failed (") +
                cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
        }
    }

    void cuda_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
        if (bytes == 0) return;
        auto stream = get_cuda_context().stream_handle();
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("velomind: cudaMemcpyAsync (D2H) failed for ") +
                std::to_string(bytes) + " bytes (" +
                cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("velomind: cudaStreamSynchronize (D2H) failed (") +
                cudaGetErrorName(err) + "): " + cudaGetErrorString(err));
        }
    }

    bool cuda_device_available() noexcept {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

    VELOMIND_REGISTER_STORAGE_CREATOR(DeviceType::CUDA, cuda_allocator)
    VELOMIND_REGISTER_MEMORY_TRANSFER(DeviceType::CUDA, cuda_memcpy_h2d, cuda_memcpy_d2h)
    VELOMIND_REGISTER_DEVICE_AVAILABILITY(DeviceType::CUDA, cuda_device_available)

}

}
