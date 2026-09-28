#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace velomind::examples::smollm2 {

// 轻量级 JSON DOM 节点，用于解析分词器和模型配置 JSON。
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type                                           type = Type::Null;
    bool                                           b    = false;
    double                                         n    = 0.0;
    std::string                                    s;
    std::vector<JsonValue>                         a;

    // 保持对象字段顺序，兼容 tokenizer.json 中依赖声明顺序的映射表
    std::vector<std::pair<std::string, JsonValue>> o;

    [[nodiscard]] auto is_object() const noexcept { return type == Type::Object; }
    [[nodiscard]] auto is_array()  const noexcept { return type == Type::Array;  }
    [[nodiscard]] auto is_string() const noexcept { return type == Type::String; }
    [[nodiscard]] auto is_number() const noexcept { return type == Type::Number; }

    [[nodiscard]] const JsonValue* find(std::string_view key) const {
        if (!is_object()) return nullptr;
        for (const auto& kv : o) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

// 轻量递归下降 JSON 解析器
class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) {
            throw std::runtime_error("json: trailing content after value");
        }
        return v;
    }

private:
    const std::string& s_;
    std::size_t        pos_;

    void skip_ws() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++pos_;
            else break;
        }
    }

    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("json: " + msg + " at offset " + std::to_string(pos_));
    }

    char peek() const {
        if (pos_ >= s_.size()) {
            throw std::runtime_error("json: unexpected end of input at offset " + std::to_string(pos_));
        }
        return s_[pos_];
    }

    char get() {
        if (pos_ >= s_.size()) {
            throw std::runtime_error("json: unexpected end of input at offset " + std::to_string(pos_));
        }
        return s_[pos_++];
    }

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end of input");
        char c = s_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    JsonValue parse_object() {
        JsonValue v; v.type = JsonValue::Type::Object;
        get();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return v; }
        while (true) {
            skip_ws();

            std::string key = parse_string_value_impl();
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') {
                fail("expected ':' after object key");
            }
            ++pos_;
            skip_ws();
            JsonValue val = parse_value();
            v.o.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (pos_ >= s_.size()) fail("unexpected end of input in object");
            char c = s_[pos_++];
            if (c == ',') continue;
            if (c == '}') return v;
            fail("expected ',' or '}' in object");
        }
    }

    JsonValue parse_array() {
        JsonValue v; v.type = JsonValue::Type::Array;
        get();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return v; }
        while (true) {
            skip_ws();
            v.a.push_back(parse_value());
            skip_ws();
            if (pos_ >= s_.size()) fail("unexpected end of input in array");
            char c = s_[pos_++];
            if (c == ',') continue;
            if (c == ']') return v;
            fail("expected ',' or ']' in array");
        }
    }

    std::string parse_string_value_impl() {
        if (pos_ >= s_.size() || s_[pos_] != '"') {
            fail("expected '\"' at start of string");
        }
        ++pos_;
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) fail("bad escape");
                char esc = s_[pos_++];
                switch (esc) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        if (pos_ + 4 > s_.size()) fail("short \\u escape");
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = s_[pos_++];
                            cp <<= 4;
                            if      (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else fail("invalid hex digit");
                        }
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xF0 | (cp >> 18));
                            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: fail(std::string("unsupported escape \\") + esc);
                }
            } else {
                out += c;
            }
        }
    }

    JsonValue parse_string_value() {
        JsonValue v; v.type = JsonValue::Type::String;
        v.s = parse_string_value_impl();
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::Type::Bool;
        if (pos_ + 4 <= s_.size() && s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4; v.b = true; return v;
        }
        if (pos_ + 5 <= s_.size() && s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5; v.b = false; return v;
        }
        fail("expected boolean");
    }

    JsonValue parse_null() {
        if (pos_ + 4 > s_.size() || s_.compare(pos_, 4, "null") != 0) {
            fail("expected 'null'");
        }
        pos_ += 4;
        return JsonValue{};
    }

    JsonValue parse_number() {
        std::size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos_;
            } else break;
        }
        if (start == pos_) fail("expected number");
        JsonValue v;
        v.n = std::stod(s_.substr(start, pos_ - start));
        v.type = JsonValue::Type::Number;
        return v;
    }
};

} // namespace velomind::examples::smollm2
