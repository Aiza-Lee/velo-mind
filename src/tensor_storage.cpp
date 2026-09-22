#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <memory>

#include "internal/registry/storage.h"

namespace velomind {

auto TensorStorage::is_available(DeviceType device) noexcept -> bool {
    return internal::is_device_available(device);
}

auto TensorStorage::allocate(std::size_t bytes, DeviceType device) -> std::shared_ptr<TensorStorage> {
    auto fn = internal::get_storage_creator(device);
    if (fn == nullptr) return nullptr;
    return fn(bytes);
}

} // namespace velomind
