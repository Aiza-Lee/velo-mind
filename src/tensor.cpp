#include "velomind/tensor.h"

namespace velomind {

Tensor::Tensor(const shape_t& shape, DataType dtype, DeviceType device) : _shape(shape), _data_type(dtype), _device_type(device) {
}

} // namespace velomind
