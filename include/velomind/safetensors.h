#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>

#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind {

// 偏移量相对于 safetensors 数据区起点，而非文件起点。
struct SafetensorsHeaderEntry {
    DataType    dtype = DataType::Float32;
    shape_t     shape;
    std::size_t begin = 0;
    std::size_t end   = 0;
};

struct SafetensorsHeader {
    std::unordered_map<std::string, SafetensorsHeaderEntry> tensors;
};

// 解析 safetensors 格式头部 JSON 字节切片，提取张量元数据表（形状、数据类型及相对数据区字节偏移），不分配物理内存。
auto parse_safetensors_header(std::span<const std::byte> header_bytes)
    -> std::optional<SafetensorsHeader>;

// safetensors 权重文件解析结果：持有所有解析成功的 Host 端物理张量存储（TensorStorage），可直接供 Graph 共享复用。
struct SafetensorsFile {
    std::unordered_map<std::string, std::shared_ptr<TensorStorage>> tensors;
};

// 从本地磁盘完整读取 safetensors 权重文件，在 Host 端分配物理存储并反序列化二进制数据，返回按张量名称索引的映射表。
auto load_safetensors(const std::filesystem::path& path)
    -> std::optional<SafetensorsFile>;

} // namespace velomind
