
#include "velomind/types.h"
#include <cstdint>
#include <stdexcept>

namespace velomind::utils {

inline auto type_bytes(DataType dtype) -> size_t {
    switch (dtype) {
    case DataType::Float32: return sizeof(float);
    case DataType::Float64: return sizeof(double);
    case DataType::Int32:   return sizeof(int32_t);
    default:
        throw std::runtime_error("Unsupported data type");
    }
}

inline auto shape_numel(const shape_t &shape) -> size_t {
    size_t numel = 1;
    for (const auto &dim : shape) {
        numel *= dim;
    }
    return numel;
}

} // namespace velomind::utils
