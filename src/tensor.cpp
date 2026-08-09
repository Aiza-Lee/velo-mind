#include <memory>

#include "velomind/tensor.h"

namespace velomind {

Tensor::Tensor(const shape_t& shape, DataType dtype, DeviceType device) 
    : _shape(shape), _data_type(dtype), _device_type(device) {
    size_t size_bytes = numel() * data_element_size();
    if (device == DeviceType::CPU) {
        _storage = std::make_unique<CpuStorage>(size_bytes);
    } else if (device == DeviceType::CUDA) {
        _storage = std::make_unique<CudaStorage>(size_bytes);
    } else {
        throw std::runtime_error("Unsupported device type");
    }
}

template<SupportedDataType T> auto Tensor::fill(T value) -> void {
    if (_device_type == DeviceType::CPU) {
        T* data_ptr = static_cast<T*>(_storage->data());
        std::fill(data_ptr, data_ptr + numel(), value);
    } else if (_device_type == DeviceType::CUDA) {
        // Implement CUDA fill logic here
        throw std::runtime_error("CUDA fill not implemented yet");
    } else {
        throw std::runtime_error("Unsupported device type");
    }
}

} // namespace velomind
