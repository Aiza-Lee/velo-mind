#include "velomind/safetensors.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "internal/log.h"

namespace velomind {

namespace {

constexpr std::size_t kMaxHeaderBytes = 100U * 1024U * 1024U;
constexpr std::size_t kMaxJsonDepth = 64;

// safetensors 头只需要对象、数组、字符串和整数，用轻量解析器避免引入 JSON 依赖。
enum class JsonType { Object, Array, String, Int };

struct JsonValue {
    JsonType                              type = JsonType::Object;
    std::string                           str;
    std::int64_t                          integer = 0;
    std::map<std::string, JsonValue>      object;
    std::vector<JsonValue>                array;
};

class JsonCursor {
public:
    explicit JsonCursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

    auto eof() const noexcept -> bool { return pos_ >= bytes_.size(); }

    auto parse_value(std::size_t depth = 0) -> JsonValue {
        if (depth > kMaxJsonDepth) throw std::runtime_error("JSON: nesting limit exceeded");
        skip_ws();
        if (eof()) throw std::runtime_error("JSON: unexpected end of input");
        std::byte c = bytes_[pos_];
        if (c == std::byte{'{'}) return parse_object(depth);
        if (c == std::byte{'['}) return parse_array(depth);
        if (c == std::byte{'"'}) return parse_string_value();
        if (c == std::byte{'-'} || (c >= std::byte{'0'} && c <= std::byte{'9'})) {
            return parse_int_value();
        }
        throw std::runtime_error(std::string{"JSON: unexpected character '"} +
                                 static_cast<char>(c) + "'");
    }

    auto at_end() -> bool { skip_ws(); return eof(); }

    auto parse_header_object() -> std::map<std::string, JsonValue> {
        JsonValue v = parse_object(0);
        return std::move(v.object);
    }

private:
    void skip_ws() noexcept {
        while (pos_ < bytes_.size()) {
            std::byte c = bytes_[pos_];
            if (c == std::byte{' '} || c == std::byte{'\t'} ||
                c == std::byte{'\n'} || c == std::byte{'\r'}) {
                ++pos_;
            } else {
                break;
            }
        }
    }

    auto peek() -> std::byte {
        if (eof()) throw std::runtime_error("JSON: unexpected end of input");
        return bytes_[pos_];
    }

    void expect(std::byte c) {
        if (eof() || bytes_[pos_] != c) {
            std::string got = eof() ? "<eof>" : std::string{1, static_cast<char>(bytes_[pos_])};
            throw std::runtime_error(std::string{"JSON: expected '"} +
                                     static_cast<char>(c) + "', got '" + got + "'");
        }
        ++pos_;
    }

    auto parse_object(std::size_t depth) -> JsonValue {
        expect(std::byte{'{'});
        JsonValue out;
        out.type = JsonType::Object;
        skip_ws();
        if (!eof() && bytes_[pos_] == std::byte{'}'}) {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(std::byte{':'});
            JsonValue val = parse_value(depth + 1);
            if (!out.object.emplace(std::move(key), std::move(val)).second)
                throw std::runtime_error("JSON: duplicate object key");
            skip_ws();
            if (eof()) throw std::runtime_error("JSON: unterminated object");
            if (bytes_[pos_] == std::byte{','}) { ++pos_; continue; }
            if (bytes_[pos_] == std::byte{'}'}) { ++pos_; break; }
            throw std::runtime_error("JSON: expected ',' or '}' in object");
        }
        return out;
    }

    auto parse_array(std::size_t depth) -> JsonValue {
        expect(std::byte{'['});
        JsonValue out;
        out.type = JsonType::Array;
        skip_ws();
        if (!eof() && bytes_[pos_] == std::byte{']'}) {
            ++pos_;
            return out;
        }
        while (true) {
            JsonValue v = parse_value(depth + 1);
            out.array.push_back(std::move(v));
            skip_ws();
            if (eof()) throw std::runtime_error("JSON: unterminated array");
            if (bytes_[pos_] == std::byte{','}) { ++pos_; continue; }
            if (bytes_[pos_] == std::byte{']'}) { ++pos_; break; }
            throw std::runtime_error("JSON: expected ',' or ']' in array");
        }
        return out;
    }

    auto parse_string() -> std::string {
        expect(std::byte{'"'});
        std::string out;
        while (true) {
            if (eof()) throw std::runtime_error("JSON: unterminated string");
            std::byte c = bytes_[pos_++];
            if (c == std::byte{'"'}) return out;
            if (c == std::byte{'\\'}) {
                if (eof()) throw std::runtime_error("JSON: bad escape");
                std::byte esc = bytes_[pos_++];
                switch (static_cast<unsigned char>(esc)) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'n':  out.push_back('\n'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'u': {
                        // JSON 的 \u 转义在此转为 UTF-8；模型键名通常落在 BMP 内。

                        if (pos_ + 4 > bytes_.size())
                            throw std::runtime_error("JSON: truncated \\u escape");
                        std::uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            std::byte h = bytes_[pos_++];
                            cp <<= 4;
                            char hc = static_cast<char>(h);
                            if      (hc >= '0' && hc <= '9') cp |= static_cast<std::uint32_t>(hc - '0');
                            else if (hc >= 'a' && hc <= 'f') cp |= static_cast<std::uint32_t>(hc - 'a' + 10);
                            else if (hc >= 'A' && hc <= 'F') cp |= static_cast<std::uint32_t>(hc - 'A' + 10);
                            else throw std::runtime_error("JSON: bad hex digit in \\u escape");
                        }
                        if (cp < 0x80) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            std::string{"JSON: unknown escape \\"} +
                            static_cast<char>(esc));
                }
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
    }

    auto parse_string_value() -> JsonValue {
        JsonValue v;
        v.type = JsonType::String;
        v.str  = parse_string();
        return v;
    }

    auto parse_int_value() -> JsonValue {
        JsonValue v;
        v.type = JsonType::Int;
        std::size_t start = pos_;
        if (!eof() && bytes_[pos_] == std::byte{'-'}) ++pos_;
        if (eof() || bytes_[pos_] < std::byte{'0'} || bytes_[pos_] > std::byte{'9'}) {
            throw std::runtime_error("JSON: expected integer");
        }
        while (!eof() && bytes_[pos_] >= std::byte{'0'} && bytes_[pos_] <= std::byte{'9'}) ++pos_;

        std::string s(reinterpret_cast<const char*>(bytes_.data() + start),
                      pos_ - start);
        v.integer = std::stoll(s);
        return v;
    }

    std::span<const std::byte> bytes_;
    std::size_t                 pos_ = 0;
};

auto map_safetensors_dtype(const std::string& s) -> std::optional<DataType> {
    if (s == "F32")  return DataType::Float32;
    if (s == "F16")  return DataType::Float16;
    if (s == "BF16") return DataType::BFloat16;
    if (s == "I32")  return DataType::Int32;
    if (s == "I8")   return DataType::Int8;
    if (s == "BOOL") return DataType::Bool;

    return std::nullopt;
}

auto safetensors_dtype_byte_size(const std::string& s) -> std::optional<std::size_t> {
    if (s == "F64")  return 8;
    if (s == "F32")  return 4;
    if (s == "F16")  return 2;
    if (s == "BF16") return 2;
    if (s == "I64")  return 8;
    if (s == "I32")  return 4;
    if (s == "I16")  return 2;
    if (s == "I8")   return 1;
    if (s == "U64")  return 8;
    if (s == "U32")  return 4;
    if (s == "U16")  return 2;
    if (s == "U8")   return 1;
    if (s == "BOOL") return 1;
    return std::nullopt;
}

constexpr const char* kMetadataKey = "__metadata__";

// 解析单个张量描述符并校验形状与字节区间一致性
auto parse_single_tensor_entry(
    const std::string& name,
    const JsonValue& val
) -> std::optional<SafetensorsHeaderEntry> {
    if (val.type != JsonType::Object) {
        VELOMIND_LOG(Error, "safetensors entry '{}' is not an object", name);
        return std::nullopt;
    }

    auto it_dtype = val.object.find("dtype");
    auto it_shape = val.object.find("shape");
    auto it_offs  = val.object.find("data_offsets");
    if (it_dtype == val.object.end() ||
        it_shape == val.object.end() ||
        it_offs  == val.object.end()) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' missing dtype/shape/data_offsets", name);
        return std::nullopt;
    }
    if (it_dtype->second.type != JsonType::String ||
        it_shape->second.type != JsonType::Array  ||
        it_offs->second.type  != JsonType::Array) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' has wrong field types", name);
        return std::nullopt;
    }

    auto dtype = map_safetensors_dtype(it_dtype->second.str);
    if (!dtype) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' has unsupported dtype '{}'",
            name, it_dtype->second.str);
        return std::nullopt;
    }

    shape_t shape;
    shape.reserve(it_shape->second.array.size());
    for (auto& dim : it_shape->second.array) {
        if (dim.type != JsonType::Int || dim.integer < 0 ||
            static_cast<std::uint64_t>(dim.integer) > std::numeric_limits<std::size_t>::max()) {
            VELOMIND_LOG(Error,
                "safetensors entry '{}' has non-integer/negative dim", name);
            return std::nullopt;
        }
        shape.push_back(static_cast<dim_t>(dim.integer));
    }
    std::size_t numel = std::any_of(shape.begin(), shape.end(), [](dim_t dim) { return dim == 0; }) ? 0 : 1;
    if (numel != 0) {
        for (auto dim : shape) {
            const auto width = static_cast<std::size_t>(dim);
            if (numel > std::numeric_limits<std::size_t>::max() / width)
                return std::nullopt;
            numel *= width;
        }
    }

    if (it_offs->second.array.size() != 2 ||
        it_offs->second.array[0].type != JsonType::Int ||
        it_offs->second.array[1].type != JsonType::Int) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' has malformed data_offsets", name);
        return std::nullopt;
    }
    std::int64_t begin = it_offs->second.array[0].integer;
    std::int64_t end   = it_offs->second.array[1].integer;
    if (begin < 0 || end < 0 || end < begin) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' has invalid data_offsets [{}, {})",
            name, begin, end);
        return std::nullopt;
    }

    auto byte_size = safetensors_dtype_byte_size(it_dtype->second.str);
    if (!byte_size) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' dtype '{}' has no byte size",
            name, it_dtype->second.str);
        return std::nullopt;
    }
    if (static_cast<std::uint64_t>(end) > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(begin) > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    std::size_t declared_bytes = static_cast<std::size_t>(end - begin);
    if (numel > std::numeric_limits<std::size_t>::max() / *byte_size)
        return std::nullopt;
    std::size_t expected_bytes = numel * (*byte_size);
    if (declared_bytes != expected_bytes) {
        VELOMIND_LOG(Error,
            "safetensors entry '{}' data_offsets span {} bytes, expected {}",
            name, declared_bytes, expected_bytes);
        return std::nullopt;
    }
    const auto ubegin = static_cast<std::size_t>(begin);
    const auto uend   = static_cast<std::size_t>(end);
    if (ubegin > std::numeric_limits<std::size_t>::max() - expected_bytes ||
        ubegin + expected_bytes != uend) {
        return std::nullopt;
    }

    SafetensorsHeaderEntry entry;
    entry.dtype = *dtype;
    entry.shape = std::move(shape);
    entry.begin = ubegin;
    entry.end   = uend;
    return entry;
}

// 校验各张量数据区间互不重叠
auto validate_tensor_ranges(
    const std::unordered_map<std::string, SafetensorsHeaderEntry>& tensors
) -> bool {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(tensors.size());
    for (const auto& [_, entry] : tensors) {
        ranges.emplace_back(entry.begin, entry.end);
    }
    std::sort(ranges.begin(), ranges.end());
    std::size_t previous_end = 0;
    for (const auto& [begin, end] : ranges) {
        if (begin < previous_end) return false;
        previous_end = std::max(previous_end, end);
    }
    return true;
}

auto parse_safetensors_header_impl(
    std::span<const std::byte> header_bytes
) -> std::optional<SafetensorsHeader> {
    if (header_bytes.empty() || header_bytes.size() > kMaxHeaderBytes) return std::nullopt;
    JsonCursor cur(header_bytes);
    JsonValue root;
    try {
        root = cur.parse_value();
        if (!cur.at_end()) throw std::runtime_error("JSON: trailing content");
    } catch (const std::exception& e) {
        VELOMIND_LOG(Error, "safetensors header JSON parse failed: {}", e.what());
        return std::nullopt;
    }
    if (root.type != JsonType::Object) {
        VELOMIND_LOG(Error, "safetensors header root is not an object");
        return std::nullopt;
    }

    SafetensorsHeader out;
    for (const auto& [name, val] : root.object) {
        if (name == kMetadataKey) continue;
        auto entry = parse_single_tensor_entry(name, val);
        if (!entry) return std::nullopt;
        out.tensors.emplace(name, std::move(*entry));
    }

    if (!validate_tensor_ranges(out.tensors)) {
        return std::nullopt;
    }
    return out;
}

// 从文件流读取 safetensors JSON 头部字节
auto read_safetensors_header_bytes(
    std::ifstream& file,
    std::streamsize total
) -> std::optional<std::vector<std::byte>> {
    if (total < static_cast<std::streamsize>(sizeof(std::uint64_t))) {
        return std::nullopt;
    }
    file.seekg(0);

    std::uint8_t header_len_le[sizeof(std::uint64_t)]{};
    file.read(reinterpret_cast<char*>(header_len_le), sizeof(header_len_le));
    if (!file || file.gcount() != sizeof(header_len_le)) {
        VELOMIND_LOG(Error, "safetensors: failed reading header length");
        return std::nullopt;
    }
    std::uint64_t header_len = 0;
    for (std::size_t i = 0; i < sizeof(header_len_le); ++i) {
        header_len |= static_cast<std::uint64_t>(header_len_le[i]) << (8 * i);
    }

    if (header_len == 0) {
        VELOMIND_LOG(Error, "safetensors: header length is zero");
        return std::nullopt;
    }
    if (header_len > kMaxHeaderBytes ||
        header_len > std::numeric_limits<std::uint64_t>::max() - sizeof(std::uint64_t) ||
        sizeof(std::uint64_t) + header_len > static_cast<std::uint64_t>(total)) {
        VELOMIND_LOG(Error,
            "safetensors: header length {} exceeds file size {}", header_len, total);
        return std::nullopt;
    }

    std::vector<std::byte> header_bytes(header_len);
    file.read(reinterpret_cast<char*>(header_bytes.data()),
              static_cast<std::streamsize>(header_len));
    if (!file || static_cast<std::uint64_t>(file.gcount()) != header_len) {
        VELOMIND_LOG(Error, "safetensors: truncated header read");
        return std::nullopt;
    }
    return header_bytes;
}

// 读取单个张量数据并分配 CPU 存储
auto read_tensor_storage_from_file(
    std::ifstream& file,
    const std::string& name,
    const SafetensorsHeaderEntry& entry,
    std::uint64_t data_start,
    std::uint64_t data_bytes
) -> std::shared_ptr<TensorStorage> {
    if (entry.begin > entry.end) return nullptr;
    std::size_t bytes = entry.end - entry.begin;
    if (entry.end > data_bytes) {
        VELOMIND_LOG(Error,
            "safetensors: tensor '{}' extends past end of file", name);
        return nullptr;
    }
    if (entry.begin > std::numeric_limits<std::uint64_t>::max() - data_start) {
        return nullptr;
    }
    std::uint64_t tensor_offset = data_start + static_cast<std::uint64_t>(entry.begin);
    if (tensor_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return nullptr;
    }

    auto storage = TensorStorage::allocate(bytes, DeviceType::CPU);
    if (!storage) {
        VELOMIND_LOG(Error,
            "safetensors: CPU allocator returned null for tensor '{}'", name);
        return nullptr;
    }
    storage->dtype      = entry.dtype;
    storage->shape      = entry.shape;
    storage->size_bytes = bytes;

    if (bytes > 0 && storage->data == nullptr) return nullptr;
    file.seekg(static_cast<std::streamoff>(tensor_offset), std::ios::beg);
    if (bytes > 0) {
        file.read(static_cast<char*>(storage->data), static_cast<std::streamsize>(bytes));
    }
    if (!file || (bytes > 0 && static_cast<std::size_t>(file.gcount()) != bytes)) {
        VELOMIND_LOG(Error,
            "safetensors: failed reading tensor '{}' ({} bytes)", name, bytes);
        return nullptr;
    }
    return storage;
}

} // namespace

auto parse_safetensors_header(
    std::span<const std::byte> header_bytes
) -> std::optional<SafetensorsHeader> {
    return parse_safetensors_header_impl(header_bytes);
}

auto load_safetensors(
    const std::filesystem::path& path
) -> std::optional<SafetensorsFile> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        VELOMIND_LOG(Error, "safetensors: cannot open '{}'", path.string());
        return std::nullopt;
    }
    std::streamsize total = file.tellg();

    auto header_bytes = read_safetensors_header_bytes(file, total);
    if (!header_bytes) {
        return std::nullopt;
    }

    auto header = parse_safetensors_header_impl(
        std::span<const std::byte>(header_bytes->data(), header_bytes->size())
    );
    if (!header) return std::nullopt;

    const auto data_start = sizeof(std::uint64_t) + header_bytes->size();
    const auto data_bytes = static_cast<std::uint64_t>(total) - data_start;

    SafetensorsFile out;
    out.tensors.reserve(header->tensors.size());

    for (const auto& [name, entry] : header->tensors) {
        auto storage = read_tensor_storage_from_file(
            file,
            name,
            entry,
            data_start,
            data_bytes
        );
        if (!storage) return std::nullopt;
        out.tensors.emplace(name, std::move(storage));
    }
    return out;
}

} // namespace velomind
