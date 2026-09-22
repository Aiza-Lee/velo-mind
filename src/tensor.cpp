#include "velomind/tensor.h"

#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>

#include "velomind/graph.h"
#include "internal/registry/memory_transfer.h"

namespace velomind {

Tensor::Tensor(Graph* graph, std::size_t index) noexcept
    : _graph(graph), _index(index),
      _graph_lifetime(graph ? graph->lifetime_token() : std::weak_ptr<void>{}) {}

auto Tensor::storage() const -> pConstTensorStorage {
    if (_graph == nullptr || _graph_lifetime.expired()) return nullptr;
    return _graph->tensor_storage_at(_index);
}

auto Tensor::storage() -> pTensorStorage {
    if (_graph == nullptr || _graph_lifetime.expired()) return nullptr;
    return _graph->tensor_storage_at(_index);
}

auto Tensor::shared_storage() const -> std::shared_ptr<TensorStorage> {
    if (_graph == nullptr || _graph_lifetime.expired()) return nullptr;
    return _graph->shared_storage_at(_index);
}

auto Tensor::shape() const -> const shape_t& {
    static const shape_t empty;
    auto* s = storage();
    return s ? s->shape : empty;
}

auto Tensor::dtype() const -> DataType {
    auto* s = storage();
    return s ? s->dtype : DataType::Float32;
}

auto Tensor::device() const -> DeviceType {
    auto* s = storage();
    return s ? s->device : DeviceType::CPU;
}

auto Tensor::numel() const -> std::size_t {
    auto* s = storage();
    return s ? storage_numel(*s) : 0;
}

auto Tensor::nbytes() const -> std::size_t {
    auto* s = storage();
    return s ? s->size_bytes : 0;
}

auto Tensor::strides() const -> stride_t {
    auto* s = storage();
    return s ? s->effective_strides() : stride_t{};
}

auto Tensor::offset_bytes() const -> std::size_t {
    auto* s = storage();
    return s ? s->offset_bytes : 0;
}

auto Tensor::capacity_bytes() const -> std::size_t {
    auto* s = storage();
    return s ? s->capacity_bytes : 0;
}

auto Tensor::is_contiguous() const -> bool {
    auto* s = storage();
    return s ? s->is_contiguous() : true;
}

auto Tensor::is_alias() const -> bool {
    auto* s = storage();
    return s ? s->is_alias() : false;
}

auto Tensor::data() const -> const void* {
    auto* s = storage();
    return s ? s->data : nullptr;
}

auto Tensor::data() -> void* {
    auto* s = storage();
    return s ? s->data : nullptr;
}

void Tensor::copy_from_host(std::span<const std::byte> data) {
    auto* s = storage();
    if (s == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_from_host: handle not bound (call Graph::build first)");
    }
    if (data.size() != s->size_bytes) {
        throw std::runtime_error("Tensor::copy_from_host: size mismatch");
    }
    copy_from_host(data, 0);
}

void Tensor::copy_from_host(std::span<const std::byte> data, std::size_t dst_offset_bytes) {
    auto* s = storage();
    if (s == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_from_host: handle not bound (call Graph::build first)");
    }
    if (dst_offset_bytes > s->size_bytes || data.size() > s->size_bytes - dst_offset_bytes) {
        throw std::runtime_error("Tensor::copy_from_host: offset and size exceed tensor bounds");
    }
    if (data.empty()) return;
    if (s->data == nullptr) {
        throw std::logic_error("Tensor::copy_from_host: tensor buffer is not allocated");
    }
    auto t = internal::get_memory_transfer(s->device);
    if (t.copy_h2d == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_from_host: no host->device transfer for device");
    }
    auto* dst = static_cast<char*>(s->data) + dst_offset_bytes;
    t.copy_h2d(dst, data.data(), data.size());
}

void Tensor::copy_to_host(std::span<std::byte> data) const {
    auto* s = storage();
    if (s == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_to_host: handle not bound (call Graph::build first)");
    }
    if (data.size() != s->size_bytes) {
        throw std::runtime_error("Tensor::copy_to_host: size mismatch");
    }
    copy_to_host(data, 0);
}

void Tensor::copy_to_host(std::span<std::byte> data, std::size_t src_offset_bytes) const {
    auto* s = storage();
    if (s == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_to_host: handle not bound (call Graph::build first)");
    }
    if (src_offset_bytes > s->size_bytes || data.size() > s->size_bytes - src_offset_bytes) {
        throw std::runtime_error("Tensor::copy_to_host: offset and size exceed tensor bounds");
    }
    if (data.empty()) return;
    if (s->data == nullptr) {
        throw std::logic_error("Tensor::copy_to_host: tensor buffer is not allocated");
    }
    auto t = internal::get_memory_transfer(s->device);
    if (t.copy_d2h == nullptr) {
        throw std::runtime_error(
            "Tensor::copy_to_host: no device->host transfer for device");
    }
    const auto* src = static_cast<const char*>(s->data) + src_offset_bytes;
    t.copy_d2h(data.data(), src, data.size());
}

namespace {

// 打印单个 CPU 数据元素
auto print_tensor_element(
    std::ostream& os,
    DataType dtype,
    const void* elem_ptr
) -> void {
    switch (dtype) {
        case DataType::Float32:
            os << *static_cast<const float*>(elem_ptr);
            break;
        case DataType::Int32:
            os << *static_cast<const std::int32_t*>(elem_ptr);
            break;
        case DataType::Int8:
            os << static_cast<int>(*static_cast<const std::int8_t*>(elem_ptr));
            break;
        case DataType::Bool:
            os << (*static_cast<const bool*>(elem_ptr) ? "true" : "false");
            break;
        case DataType::Float16:
            os << static_cast<float>(*static_cast<const float16_t*>(elem_ptr));
            break;
        case DataType::BFloat16:
            os << static_cast<float>(*static_cast<const bfloat16_t*>(elem_ptr));
            break;
    }
}

// 格式化 CPU 张量数据预览
auto format_cpu_tensor_preview(
    std::ostream& ss,
    const TensorStorage& s,
    std::size_t n,
    bool print_data
) -> void {
    ss << ", data=[";
    const auto* base_ptr = static_cast<const std::byte*>(s.data) + s.offset_bytes;
    const auto eff_strides = s.effective_strides();
    const auto elem_size = data_type_size(s.dtype);

    auto print_at_offset = [&](std::size_t offset_in_elems) {
        const void* elem_ptr = base_ptr + offset_in_elems * elem_size;
        print_tensor_element(ss, s.dtype, elem_ptr);
    };

    auto get_linear_offset = [&](std::size_t flat_idx) -> std::size_t {
        if (s.is_contiguous() || s.shape.empty()) return flat_idx;
        std::size_t offset = 0;
        std::size_t rem = flat_idx;
        for (std::size_t i = s.shape.size(); i-- > 0;) {
            dim_t d = s.shape[i];
            if (d <= 0) continue;
            std::size_t coord = rem % d;
            rem /= d;
            offset += coord * eff_strides[i];
        }
        return offset;
    };

    const std::size_t max_show = (print_data ? 32 : 8);
    if (n <= max_show) {
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) ss << ", ";
            print_at_offset(get_linear_offset(i));
        }
    } else {
        for (std::size_t i = 0; i < 3; ++i) {
            if (i > 0) ss << ", ";
            print_at_offset(get_linear_offset(i));
        }
        ss << ", ...";
        for (std::size_t i = n - 3; i < n; ++i) {
            ss << ", ";
            print_at_offset(get_linear_offset(i));
        }
    }
    ss << "]";
}

} // namespace

auto Tensor::to_string(bool print_data) const -> std::string {
    if (_graph == nullptr || _graph_lifetime.expired()) {
        return "Tensor(invalid or expired)";
    }
    auto* s = storage();
    if (s == nullptr) {
        return "Tensor(uninitialized)";
    }

    std::ostringstream ss;
    ss << "Tensor(shape=" << velomind::to_string(s->shape)
       << ", dtype=" << data_type_name(s->dtype)
       << ", device=" << device_type_name(s->device);

    if (!s->strides.empty()) {
        ss << ", strides=" << velomind::to_string(s->strides);
    }
    if (s->offset_bytes > 0) {
        ss << ", offset_bytes=" << s->offset_bytes;
    }
    if (s->is_alias()) {
        ss << ", alias=true";
    }

    if (s->data == nullptr) {
        ss << ", unallocated)";
        return ss.str();
    }

    const std::size_t n = numel();
    const bool should_preview = print_data || (s->device == DeviceType::CPU && n <= 8);

    if (should_preview) {
        if (s->device != DeviceType::CPU) {
            ss << ", device_data=" << s->data;
        } else {
            format_cpu_tensor_preview(ss, *s, n, print_data);
        }
    }

    ss << ")";
    return ss.str();
}

} // namespace velomind
