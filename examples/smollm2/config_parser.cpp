#include "model.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace velomind::examples::smollm2 {

namespace {

    struct Parser {
        std::string_view text;
        std::size_t      pos = 0;

        void skip_ws() {
            while (pos < text.size() &&
                   (text[pos] == ' ' || text[pos] == '\t' ||
                    text[pos] == '\n' || text[pos] == '\r')) {
                ++pos;
            }
        }

        std::string read_string() {
            if (pos >= text.size() || text[pos] != '"') {
                throw std::runtime_error(
                    "smollm2::parse_config_json: expected '\"'");
            }
            ++pos;
            std::string out;
            while (pos < text.size() && text[pos] != '"') {
                if (text[pos] == '\\' && pos + 1 < text.size()) {
                    char esc = text[pos + 1];
                    switch (esc) {
                        case 'n':  out.push_back('\n'); break;
                        case 't':  out.push_back('\t'); break;
                        case 'r':  out.push_back('\r'); break;
                        case '"':  out.push_back('"');  break;
                        case '\\': out.push_back('\\'); break;
                        case '/':  out.push_back('/');  break;
                        default:   out.push_back(esc);  break;
                    }
                    pos += 2;
                } else {
                    out.push_back(text[pos]);
                    ++pos;
                }
            }
            if (pos >= text.size()) {
                throw std::runtime_error(
                    "smollm2::parse_config_json: unterminated string");
            }
            ++pos;
            return out;
        }

        double read_number() {
            std::size_t start = pos;
            if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) ++pos;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
            if (pos < text.size() && text[pos] == '.') {
                ++pos;
                while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
            }
            if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
                ++pos;
                if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
                while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
            }
            return std::stod(std::string(text.substr(start, pos - start)));
        }

        bool read_bool() {
            if (text.substr(pos, 4) == "true")  { pos += 4; return true;  }
            if (text.substr(pos, 5) == "false") { pos += 5; return false; }
            throw std::runtime_error(
                "smollm2::parse_config_json: expected boolean");
        }

        void skip_value() {
            skip_ws();
            if (pos >= text.size()) {
                throw std::runtime_error(
                    "smollm2::parse_config_json: unexpected end of input");
            }
            char c = text[pos];
            if (c == '{' || c == '[') {
                char open = c;
                char close = (c == '{') ? '}' : ']';

                int depth = 1;
                bool in_str = false;
                ++pos;
                while (pos < text.size() && depth > 0) {
                    char ch = text[pos];
                    if (in_str) {
                        if (ch == '\\' && pos + 1 < text.size()) { pos += 2; continue; }
                        if (ch == '"') in_str = false;
                        ++pos;
                        continue;
                    }
                    if (ch == '"') { in_str = true; ++pos; continue; }
                    if (ch == open) { ++depth; ++pos; continue; }
                    if (ch == close) { --depth; ++pos; continue; }
                    ++pos;
                }
                if (depth != 0) {
                    throw std::runtime_error(
                        "smollm2::parse_config_json: unbalanced brackets");
                }
                return;
            }
            if (c == '"') { read_string(); return; }
            if (c == 't' || c == 'f') { read_bool(); return; }
            if (c == 'n') {
                if (text.substr(pos, 4) != "null") {
                    throw std::runtime_error(
                        "smollm2::parse_config_json: expected null");
                }
                pos += 4;
                return;
            }

            while (pos < text.size() &&
                   text[pos] != ',' && text[pos] != '}' && text[pos] != ']' &&
                   text[pos] != ' '  && text[pos] != '\n' && text[pos] != '\r' &&
                   text[pos] != '\t') {
                ++pos;
            }
        }
    };

    std::string find_top_level_value(std::string_view text,
                                     std::string_view key) {
        Parser p{text, 0};
        p.skip_ws();
        if (p.pos >= text.size() || text[p.pos] != '{') {
            throw std::runtime_error(
                "smollm2::parse_config_json: expected '{' at top of file");
        }
        ++p.pos;

        while (true) {
            p.skip_ws();
            if (p.pos >= text.size()) {
                throw std::runtime_error(
                    "smollm2::parse_config_json: unexpected end inside object");
            }
            if (text[p.pos] == '}') return {};

            std::size_t key_start = p.pos;
            std::string k = p.read_string();
            p.skip_ws();
            if (p.pos >= text.size() || text[p.pos] != ':') {
                throw std::runtime_error(
                    "smollm2::parse_config_json: expected ':' after key");
            }
            ++p.pos;
            p.skip_ws();

            if (k == key) {
                std::size_t val_start = p.pos;
                p.skip_value();
                std::size_t val_end = p.pos;
                return std::string(text.substr(val_start, val_end - val_start));
            }

            p.skip_value();
            p.skip_ws();
            if (p.pos < text.size() && text[p.pos] == ',') {
                ++p.pos;
                continue;
            }

            (void)key_start;
        }
    }

}

auto parse_config_json(const std::string& json_text) -> Config {
    Config cfg = kSmolLM2_135M;

    auto require_double = [&](const char* key) -> double {
        auto v = find_top_level_value(json_text, key);
        if (v.empty()) {
            throw std::runtime_error(
                std::string("smollm2::parse_config_json: missing required key '") + key + "'");
        }
        Parser p{std::string_view(v), 0};
        p.skip_ws();
        double d = p.read_number();
        return d;
    };

    auto optional_bool = [&](const char* key, bool fallback) -> bool {
        auto v = find_top_level_value(json_text, key);
        if (v.empty()) return fallback;
        Parser p{std::string_view(v), 0};
        p.skip_ws();
        return p.read_bool();
    };

    cfg.vocab_size        = static_cast<std::size_t>(require_double("vocab_size"));
    cfg.hidden_size       = static_cast<std::size_t>(require_double("hidden_size"));
    cfg.intermediate_size = static_cast<std::size_t>(require_double("intermediate_size"));
    cfg.num_layers        = static_cast<std::size_t>(require_double("num_hidden_layers"));
    cfg.num_heads         = static_cast<std::size_t>(require_double("num_attention_heads"));
    cfg.num_kv_heads      = static_cast<std::size_t>(require_double("num_key_value_heads"));
    cfg.rms_norm_eps      = static_cast<float>(require_double("rms_norm_eps"));
    cfg.rope_theta        = static_cast<float>(require_double("rope_theta"));
    cfg.tie_word_embeddings = optional_bool("tie_word_embeddings", cfg.tie_word_embeddings);

    return cfg;
}

}
