#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "velomind/safetensors.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "test_helpers.h"

namespace {

struct BuiltTensor {
    std::string            name;
    std::string            dtype_str;
    std::vector<int64_t>   shape;
    std::vector<std::byte> bytes;
};

struct BuiltFile {
    std::vector<BuiltTensor> tensors;
    std::vector<std::pair<std::string, std::string>> metadata;

    auto build() const -> std::vector<std::byte> {

        std::vector<std::pair<std::size_t, std::size_t>> offsets;
        std::size_t cursor = 0;
        for (const auto& t : tensors) {
            offsets.emplace_back(cursor, cursor + t.bytes.size());
            cursor += t.bytes.size();
        }
        std::size_t data_bytes = cursor;

        std::string json = "{";
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            if (i > 0) json.push_back(',');
            json.push_back('"');
            json += t.name;
            json += "\":{\"dtype\":\"";
            json += t.dtype_str;
            json += "\",\"shape\":[";
            for (std::size_t d = 0; d < t.shape.size(); ++d) {
                if (d > 0) json.push_back(',');
                json += std::to_string(t.shape[d]);
            }
            json += "],\"data_offsets\":[";
            json += std::to_string(offsets[i].first);
            json.push_back(',');
            json += std::to_string(offsets[i].second);
            json += "]}";
        }
        if (!metadata.empty()) {
            if (!tensors.empty()) json.push_back(',');
            json += "\"__metadata__\":{";
            for (std::size_t i = 0; i < metadata.size(); ++i) {
                if (i > 0) json.push_back(',');
                json.push_back('"');
                json += metadata[i].first;
                json += "\":\"";
                json += metadata[i].second;
                json += "\"";
            }
            json.push_back('}');
        }
        json.push_back('}');

        std::uint64_t header_len = json.size();
        std::vector<std::byte> out;
        out.resize(sizeof(header_len) + header_len + data_bytes);
        std::memcpy(out.data(), &header_len, sizeof(header_len));
        std::memcpy(out.data() + sizeof(header_len),
                    json.data(), header_len);
        std::byte* dst = out.data() + sizeof(header_len) + header_len;
        for (const auto& t : tensors) {
            std::memcpy(dst, t.bytes.data(), t.bytes.size());
            dst += t.bytes.size();
        }
        return out;
    }

    auto write_to_tmp(const std::filesystem::path& path) const -> bool {
        auto bytes = build();
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(f);
    }
};

template <typename T>
auto bytes_of(const std::vector<T>& v) -> std::vector<std::byte> {
    std::vector<std::byte> out(v.size() * sizeof(T));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::random_device rd;
        std::mt19937_64 g(rd());
        auto base = std::filesystem::temp_directory_path() /
                    ("velomind_test_" + std::to_string(g()));
        std::filesystem::create_directories(base);
        path = base;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

}

TEST_CASE("parse_safetensors_header — single F32 tensor", "[safetensors]") {
    using namespace velomind;

    BuiltFile file;
    BuiltTensor t;
    t.name      = "weight";
    t.dtype_str = "F32";
    t.shape     = {2, 3};
    t.bytes     = bytes_of<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    file.tensors.push_back(std::move(t));

    auto img = file.build();

    auto header_bytes = std::span<const std::byte>(
        img.data() + sizeof(std::uint64_t), img.size() - sizeof(std::uint64_t));

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, img.data(), sizeof(header_len));
    header_bytes = std::span<const std::byte>(
        img.data() + sizeof(std::uint64_t), header_len);

    auto header = parse_safetensors_header(header_bytes);
    REQUIRE(header.has_value());
    REQUIRE(header->tensors.size() == 1);
    auto it = header->tensors.find("weight");
    REQUIRE(it != header->tensors.end());
    REQUIRE(it->second.dtype == DataType::Float32);
    REQUIRE(it->second.shape == shape_t{2, 3});
    REQUIRE(it->second.begin == 0);
    REQUIRE(it->second.end   == 24);
}

TEST_CASE("parse_safetensors_header — multiple tensors + metadata", "[safetensors]") {
    using namespace velomind;

    BuiltFile file;
    BuiltTensor a;
    a.name = "a"; a.dtype_str = "F16"; a.shape = {4};
    a.bytes = bytes_of<uint16_t>({0x3C00, 0x4000, 0x4200, 0x4400});
    file.tensors.push_back(std::move(a));
    BuiltTensor b;
    b.name = "b"; b.dtype_str = "I32"; b.shape = {2};
    b.bytes = bytes_of<int32_t>({10, 20});
    file.tensors.push_back(std::move(b));
    file.metadata.emplace_back("format", "pt");
    file.metadata.emplace_back("producer", "test");

    auto img = file.build();
    std::uint64_t header_len = 0;
    std::memcpy(&header_len, img.data(), sizeof(header_len));
    auto header_bytes = std::span<const std::byte>(
        img.data() + sizeof(std::uint64_t), header_len);

    auto header = parse_safetensors_header(header_bytes);
    REQUIRE(header.has_value());
    REQUIRE(header->tensors.size() == 2);

    auto it_a = header->tensors.find("a");
    REQUIRE(it_a != header->tensors.end());
    REQUIRE(it_a->second.dtype == DataType::Float16);
    REQUIRE(it_a->second.shape == shape_t{4});
    REQUIRE(it_a->second.begin == 0);
    REQUIRE(it_a->second.end   == 8);

    auto it_b = header->tensors.find("b");
    REQUIRE(it_b != header->tensors.end());
    REQUIRE(it_b->second.dtype == DataType::Int32);
    REQUIRE(it_b->second.shape == shape_t{2});
    REQUIRE(it_b->second.begin == 8);
    REQUIRE(it_b->second.end   == 16);
}

TEST_CASE("parse_safetensors_header — malformed JSON returns nullopt", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    auto bad = std::as_bytes(std::span<const char>("not json at all", 16));
    auto header = parse_safetensors_header(bad);
    REQUIRE(!header.has_value());
}

TEST_CASE("parse_safetensors_header — unknown dtype returns nullopt", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    std::string json = R"({"x":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}})";
    auto bytes = std::as_bytes(std::span<const char>(json.data(), json.size()));
    auto header = parse_safetensors_header(bytes);
    REQUIRE(!header.has_value());
}

TEST_CASE("parse_safetensors_header — data_offsets byte-size mismatch returns nullopt",
          "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    std::string json = R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,100]}})";
    auto bytes = std::as_bytes(std::span<const char>(json.data(), json.size()));
    auto header = parse_safetensors_header(bytes);
    REQUIRE(!header.has_value());
}

TEST_CASE("parse_safetensors_header — rejects integer overflow and overlapping data", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;
    const std::vector<std::string> invalid_headers{
        R"({"x":{"dtype":"F32","shape":[9223372036854775807,9223372036854775807],"data_offsets":[0,4]}})",
        R"({"x":{"dtype":"F32","shape":[9223372036854775807],"data_offsets":[0,4]}})",
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})"
    };
    for (const auto& json : invalid_headers)
        REQUIRE_FALSE(parse_safetensors_header(std::as_bytes(std::span(json))).has_value());
}

TEST_CASE("parse_safetensors_header — rejects duplicate keys and deep JSON", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;
    const std::vector<std::string> invalid_headers{
        R"({"x":{"dtype":"F32","dtype":"I32","shape":[1],"data_offsets":[0,4]}})",
        R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})"
    };
    for (const auto& json : invalid_headers)
        REQUIRE_FALSE(parse_safetensors_header(std::as_bytes(std::span(json))).has_value());
    const auto deeply_nested = std::string("{\"__metadata__\":") + std::string(70, '[') +
        "0" + std::string(70, ']') + "}";
    REQUIRE_FALSE(parse_safetensors_header(std::as_bytes(std::span(deeply_nested))).has_value());
}

TEST_CASE("load_safetensors — round-trip F32 values", "[safetensors]") {
    using namespace velomind;

    TempDir tmp;
    auto path = tmp.path / "f32.safetensors";

    BuiltFile file;
    BuiltTensor t;
    t.name = "weight";
    t.dtype_str = "F32";
    t.shape = {2, 3};
    t.bytes = bytes_of<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    file.tensors.push_back(std::move(t));
    REQUIRE(file.write_to_tmp(path));

    auto loaded = load_safetensors(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->tensors.size() == 1);
    auto it = loaded->tensors.find("weight");
    REQUIRE(it != loaded->tensors.end());
    auto& s = *it->second;
    REQUIRE(s.dtype == DataType::Float32);
    REQUIRE(s.shape == shape_t{2, 3});
    REQUIRE(s.size_bytes == 24);

    std::vector<float> values(6);
    std::memcpy(values.data(), s.data, s.size_bytes);
    REQUIRE(values == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST_CASE("load_safetensors — round-trip F16 / BF16 / I32 / I8 / Bool", "[safetensors]") {
    using namespace velomind;

    TempDir tmp;

    auto run_one = [&](const std::string& dtype_str,
                       std::vector<int64_t> shape,
                       std::vector<std::byte> bytes,
                       velomind::DataType expected_dtype) {
        auto path = tmp.path / ("mixed_" + dtype_str + ".safetensors");
        BuiltFile file;
        BuiltTensor t;
        t.name = dtype_str; t.dtype_str = dtype_str; t.shape = shape;
        t.bytes = std::move(bytes);
        file.tensors.push_back(std::move(t));
        REQUIRE(file.write_to_tmp(path));

        auto loaded = load_safetensors(path);
        REQUIRE(loaded.has_value());
        auto it = loaded->tensors.find(dtype_str);
        REQUIRE(it != loaded->tensors.end());
        REQUIRE(it->second->dtype == expected_dtype);
        REQUIRE(it->second->shape == shape);
    };

    run_one("F16",  {3},
            bytes_of<std::uint16_t>({0x3C00, 0xC000, 0x4300}),
            velomind::DataType::Float16);

    run_one("BF16", {2},
            bytes_of<std::uint16_t>({0x3F80, 0xC000}),
            velomind::DataType::BFloat16);

    run_one("I32",  {4},
            bytes_of<std::int32_t>({-1, 0, 1, 2147483647}),
            velomind::DataType::Int32);

    run_one("I8",   {3},
            bytes_of<std::int8_t>({-128, 0, 127}),
            velomind::DataType::Int8);

    run_one("BOOL", {4},
            std::vector<std::byte>{std::byte{0x01}, std::byte{0x00},
                                   std::byte{0x01}, std::byte{0x00}},
            velomind::DataType::Bool);
}

TEST_CASE("load_safetensors — missing file returns nullopt", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    auto loaded = load_safetensors("/nonexistent/path/does_not_exist.safetensors");
    REQUIRE(!loaded.has_value());
}

TEST_CASE("load_safetensors — truncated file returns nullopt", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    TempDir tmp;
    auto path = tmp.path / "truncated.safetensors";

    std::uint64_t lie_len = 1000;
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&lie_len), sizeof(lie_len));
    f.close();

    auto loaded = load_safetensors(path);
    REQUIRE(!loaded.has_value());
}

TEST_CASE("load_safetensors — huge header length and offset are rejected", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;
    TempDir tmp;
    auto path = tmp.path / "oversized.safetensors";
    {
        std::uint64_t length = std::numeric_limits<std::uint64_t>::max();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    }
    REQUIRE_FALSE(load_safetensors(path).has_value());

    const std::string json = R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[9223372036854775800,9223372036854775804]}})";
    {
        std::uint64_t length = json.size();
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(json.data(), static_cast<std::streamsize>(json.size()));
    }
    REQUIRE_FALSE(load_safetensors(path).has_value());
}

TEST_CASE("load_safetensors — multiple tensors with mixed dtype + metadata",
          "[safetensors]") {
    using namespace velomind;

    TempDir tmp;
    auto path = tmp.path / "multi.safetensors";

    BuiltFile file;
    BuiltTensor a;
    a.name = "embed"; a.dtype_str = "F32"; a.shape = {4};
    a.bytes = bytes_of<float>({0.1f, 0.2f, 0.3f, 0.4f});
    file.tensors.push_back(std::move(a));
    BuiltTensor b;
    b.name = "mask"; b.dtype_str = "BOOL"; b.shape = {2, 2};
    b.bytes = std::vector<std::byte>{std::byte{0x01}, std::byte{0x00},
                                     std::byte{0x00}, std::byte{0x01}};
    file.tensors.push_back(std::move(b));
    file.metadata.emplace_back("format", "pt");
    REQUIRE(file.write_to_tmp(path));

    auto loaded = load_safetensors(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->tensors.size() == 2);

    REQUIRE(loaded->tensors.count("embed") == 1);
    REQUIRE(loaded->tensors.count("mask")  == 1);
    REQUIRE(loaded->tensors.at("embed")->dtype == DataType::Float32);
    REQUIRE(loaded->tensors.at("mask")->dtype  == DataType::Bool);
    REQUIRE(loaded->tensors.at("mask")->shape  == shape_t{2, 2});

    auto* p = static_cast<std::byte*>(loaded->tensors.at("mask")->data);
    REQUIRE(static_cast<unsigned int>(p[0]) == 1);
    REQUIRE(static_cast<unsigned int>(p[1]) == 0);
    REQUIRE(static_cast<unsigned int>(p[2]) == 0);
    REQUIRE(static_cast<unsigned int>(p[3]) == 1);
}

TEST_CASE("load_safetensors — F64 dtype is rejected (no v1 mapping)", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    TempDir tmp;
    auto path = tmp.path / "f64.safetensors";

    BuiltFile file;
    BuiltTensor t;
    t.name = "x"; t.dtype_str = "F64"; t.shape = {2};
    t.bytes = std::vector<std::byte>(16, std::byte{0});
    file.tensors.push_back(std::move(t));
    REQUIRE(file.write_to_tmp(path));

    auto loaded = load_safetensors(path);
    REQUIRE(!loaded.has_value());
}

TEST_CASE("parse_safetensors_header — comprehensive corpus tests", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;

    auto to_bytes = [](std::string_view sv) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(sv.data()), sv.size());
    };

    SECTION("truncated JSON inputs return nullopt") {
        const std::vector<std::string> truncated_cases{
            "",
            "{",
            "{\"x\":",
            "{\"x\":{",
            "{\"x\":{\"dtype\":",
            "{\"x\":{\"dtype\":\"",
            "{\"x\":{\"dtype\":\"F",
            "{\"x\":{\"dtype\":\"\\u00",
            "{\"x\":{\"dtype\":\"F32\",",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]"
        };
        for (const auto& json : truncated_cases) {
            REQUIRE_FALSE(parse_safetensors_header(to_bytes(json)).has_value());
        }
    }

    SECTION("huge dimensions, negative dimensions and numel overflow return nullopt") {
        const std::vector<std::string> overflow_cases{
            // 巨维度连乘溢出
            R"({"x":{"dtype":"F32","shape":[2000000000,2000000000,2000000000],"data_offsets":[0,4]}})",
            // 元素数在 size_t 内但乘以 byte_size 溢出
            R"({"x":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,4]}})",
            // 负维度
            R"({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
            // 非整数维度
            R"({"x":{"dtype":"F32","shape":["1"],"data_offsets":[0,4]}})",
            R"({"x":{"dtype":"F32","shape":[{}],"data_offsets":[0,4]}})",
            // 负偏移量或 begin > end
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[-1,4]}})",
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,2]}})",
            // 非整数偏移量
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":["0",4]}})",
            // 偏移量数组长度异常
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}})",
            R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}})"
        };
        for (const auto& json : overflow_cases) {
            REQUIRE_FALSE(parse_safetensors_header(to_bytes(json)).has_value());
        }
    }

    SECTION("duplicate keys in tensors and metadata return nullopt") {
        const std::vector<std::string> duplicate_cases{
            R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})",
            R"({"x":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
            R"({"x":{"shape":[1],"shape":[1],"dtype":"F32","data_offsets":[0,4]}})",
            R"({"__metadata__":{"k":"1","k":"2"}})",
            R"({"__metadata__":{"obj":{"sub":"1","sub":"2"}}})"
        };
        for (const auto& json : duplicate_cases) {
            REQUIRE_FALSE(parse_safetensors_header(to_bytes(json)).has_value());
        }
    }

    SECTION("anomalous nesting and types return nullopt") {
        // 超过 64 层的对象嵌套
        std::string deep_obj = "{\"__metadata__\":";
        for (int i = 0; i < 70; ++i) deep_obj += "{\"k\":";
        deep_obj += "\"v\"";
        for (int i = 0; i < 70; ++i) deep_obj += "}";
        deep_obj += "}";
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(deep_obj)).has_value());

        // 根非对象
        REQUIRE_FALSE(parse_safetensors_header(to_bytes("[]")).has_value());
        REQUIRE_FALSE(parse_safetensors_header(to_bytes("123")).has_value());
        REQUIRE_FALSE(parse_safetensors_header(to_bytes("\"string\"")).has_value());

        // 张量描述项非对象
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(R"({"x":"string"})")).has_value());
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(R"({"x":123})")).has_value());
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(R"({"x":[]})")).has_value());
    }

    SECTION("overlapping data ranges return nullopt and adjacent ranges succeed") {
        // 部分重叠
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(
            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})"
        )).has_value());

        // 完全包含
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(
            R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})"
        )).has_value());

        // 反向声明包含
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(
            R"({"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})"
        )).has_value());

        // 完全相同区间
        REQUIRE_FALSE(parse_safetensors_header(to_bytes(
            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})"
        )).has_value());

        // 紧邻不重叠区间成功
        auto ok_header = parse_safetensors_header(to_bytes(
            R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},"c":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})"
        ));
        REQUIRE(ok_header.has_value());
        REQUIRE(ok_header->tensors.size() == 3);
    }
}

TEST_CASE("load_safetensors — truncation at various stages returns nullopt", "[safetensors]") {
    using namespace velomind;
    velomind_test::ScopedLogSilencer silencer;
    TempDir tmp;

    SECTION("files shorter than 8 bytes") {
        for (std::size_t len : {0UL, 1UL, 4UL, 7UL}) {
            auto path = tmp.path / ("short_" + std::to_string(len) + ".safetensors");
            std::vector<char> buf(len, 0);
            std::ofstream f(path, std::ios::binary);
            if (len > 0) f.write(buf.data(), static_cast<std::streamsize>(len));
            f.close();
            REQUIRE_FALSE(load_safetensors(path).has_value());
        }
    }

    SECTION("file truncated inside header body") {
        auto path = tmp.path / "trunc_header.safetensors";
        std::uint64_t declared_len = 100;
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&declared_len), sizeof(declared_len));
        std::string partial = "{\"x\":{\"d";
        f.write(partial.data(), static_cast<std::streamsize>(partial.size()));
        f.close();
        REQUIRE_FALSE(load_safetensors(path).has_value());
    }

    SECTION("file truncated inside tensor payload") {
        auto path = tmp.path / "trunc_payload.safetensors";
        std::string json = R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
        std::uint64_t header_len = json.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        f.write(json.data(), static_cast<std::streamsize>(header_len));
        // 期望 8 字节数据，但仅写入 4 字节
        float val = 1.0f;
        f.write(reinterpret_cast<const char*>(&val), sizeof(val));
        f.close();
        REQUIRE_FALSE(load_safetensors(path).has_value());
    }
}
