#pragma once

#include <cstdint>
#include <vector>

namespace velomind {
    
using index_t = std::int64_t;
using dim_t = std::int64_t;
using shape_t = std::vector<dim_t>;

enum class DataType { Float32, Float64, Int32 };
enum class DeviceType { CPU, CUDA };

template<typename T>
concept SupportedDataType = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t>;

} // namespace velomind

