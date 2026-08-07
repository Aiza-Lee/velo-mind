#pragma once

#include <cstdint>
#include <vector>

namespace velomind {
    
using index_t = std::int64_t;
using dim_t = std::int64_t;
using shape_t = std::vector<dim_t>;

enum class DataType { Float32, Float64, Int32 };
enum class DeviceType { CPU, CUDA };

} // namespace velomind

