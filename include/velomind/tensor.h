#pragma once 

#include <memory>
#include <string>
#include <vector>

#include "velomind_core_export.h"
#include "velomind/types.h"
#include "velomind/storage.h"
#include "velomind/utils.h"

namespace velomind {

class VELOMIND_CORE_EXPORT Tensor {
public:
    Tensor(const shape_t& shape, DataType dtype = DataType::Float32, DeviceType device = DeviceType::CPU);
    ~Tensor() = default;

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    auto shape() const -> const shape_t& { return _shape; }
    auto data_type() const -> const DataType& { return _data_type; }
    auto device_type() const -> const DeviceType& { return _device_type; }

    size_t numel() const { return utils::shape_numel(_shape); }
    size_t data_element_size() const { return utils::type_bytes(_data_type); }
    size_t size_bytes() const { return _storage->size_bytes(); }

    template <SupportedDataType T> T* data() { return static_cast<T*>(_storage->data()); }
    template <SupportedDataType T> const T* data() const { return static_cast<const T*>(_storage->data()); }

	template<SupportedDataType T> auto fill(T value) -> void;
    auto to(const DeviceType& device) const -> Tensor;

    auto summary() const -> std::string;
    auto to_string() const -> std::string;

private:
    shape_t _shape;
    DataType _data_type;
    DeviceType _device_type;

    std::shared_ptr<IStorage> _storage;

};

} // namespace velomind
