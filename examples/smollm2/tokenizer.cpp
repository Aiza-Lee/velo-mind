#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace velomind::examples::smollm2 {

namespace {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type                                  type = Type::Null;
    bool                                  b    = false;
    double                                n    = 0.0;
    std::string                           s;
    std::vector<JsonValue>                a;

    // 保持对象字段顺序，兼容 tokenizer.json 中依赖声明顺序的表。
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
        throw std::runtime_error("json: " + msg + " at offset "
                                 + std::to_string(pos_));
    }

    char peek() const {
        if (pos_ >= s_.size()) {
            throw std::runtime_error("json: unexpected end of input at offset "
                                     + std::to_string(pos_));
        }
        return s_[pos_];
    }
    char get() {
        if (pos_ >= s_.size()) {
            throw std::runtime_error("json: unexpected end of input at offset "
                                     + std::to_string(pos_));
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

auto load_json_file(const std::string& path) -> JsonValue {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer: cannot open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();

    auto text = ss.str();
    JsonParser parser(text);
    return parser.parse();
}

void init_byte_to_unicode(int (&out)[256], std::vector<std::uint32_t>& rev) {
    rev.assign(0x110000, HuggingFaceBpeTokenizer::kUnmapped);
    auto char_to_byte = [&](std::uint32_t cp, std::uint8_t b) {
        rev[cp] = b;
    };

    std::vector<std::uint8_t> printable;
    for (unsigned c = 0x21; c <= 0x7E; ++c) printable.push_back(static_cast<std::uint8_t>(c));
    for (unsigned c = 0xA1; c <= 0xAC; ++c) printable.push_back(static_cast<std::uint8_t>(c));
    for (unsigned c = 0xAE; c <= 0xFF; ++c) printable.push_back(static_cast<std::uint8_t>(c));

    auto is_printable = [&](std::uint8_t b) -> bool {
        for (auto x : printable) {
            if (x == b) return true;
        }
        return false;
    };

    std::uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(static_cast<std::uint8_t>(b))) {
            out[b] = b;
        } else {
            out[b] = 256 + n;
            ++n;
        }
    }

    for (std::uint32_t b = 0; b < 256; ++b) {
        char_to_byte(static_cast<std::uint32_t>(out[b]),
                     static_cast<std::uint8_t>(b));
    }
}

}

HuggingFaceBpeTokenizer::HuggingFaceBpeTokenizer() {
    init_byte_to_unicode(byte_to_unicode_, unicode_to_byte_);
}

void HuggingFaceBpeTokenizer::load(const std::string& model_path) {
    auto root = load_json_file(model_path);
    if (!root.is_object()) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer: tokenizer.json root is not an object");
    }

    const JsonValue* model_v = root.find("model");
    if (model_v == nullptr || !model_v->is_object()) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer: missing 'model' object");
    }
    const JsonValue* vocab_v = model_v->find("vocab");
    if (vocab_v == nullptr || !vocab_v->is_object()) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer: missing model.vocab object");
    }

    std::int32_t max_id = -1;
    for (const auto& kv : vocab_v->o) {
        if (!kv.second.is_number()) continue;
        std::int32_t id = static_cast<std::int32_t>(kv.second.n);
        if (id > max_id) max_id = id;
    }
    if (max_id < 0) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer: empty vocab");
    }

    id_to_token_.assign(static_cast<std::size_t>(max_id + 1), std::string{});
    token_to_id_.clear();
    token_to_id_.reserve(vocab_v->o.size());

    for (const auto& kv : vocab_v->o) {
        if (!kv.second.is_number()) continue;
        std::int32_t id = static_cast<std::int32_t>(kv.second.n);
        id_to_token_[static_cast<std::size_t>(id)] = kv.first;
        token_to_id_.emplace(kv.first, id);
    }

    const JsonValue* merges_v = model_v->find("merges");
    if (merges_v != nullptr && merges_v->is_array()) {
        merges_.clear();
        merges_.reserve(merges_v->a.size());
        for (const auto& m : merges_v->a) {
            if (!m.is_string()) continue;
            const std::string& s = m.s;
            auto sp = s.find(' ');
            if (sp == std::string::npos) {

                merges_.emplace_back(s, std::string{});
            } else {
                merges_.emplace_back(s.substr(0, sp), s.substr(sp + 1));
            }
        }
    }

    const JsonValue* added_v = root.find("added_tokens");
    if (added_v != nullptr && added_v->is_array()) {
        for (const auto& at : added_v->a) {
            if (!at.is_object()) continue;
            const JsonValue* id_v  = at.find("id");
            const JsonValue* con_v = at.find("content");
            if (id_v == nullptr || !id_v->is_number() ||
                con_v == nullptr || !con_v->is_string()) {
                continue;
            }
            std::int32_t id = static_cast<std::int32_t>(id_v->n);
            const std::string& tok = con_v->s;

            if (static_cast<std::size_t>(id) >= id_to_token_.size()) {
                id_to_token_.resize(static_cast<std::size_t>(id) + 1);
            }
            id_to_token_[static_cast<std::size_t>(id)] = tok;
            token_to_id_[tok] = id;

            if (tok == "<|endoftext|>" || tok == "<s>" || tok == "<bos>") {
                bos_id_ = id;
                if (eos_id_ < 0) eos_id_ = id;
            } else if (tok == "<|im_end|>" || tok == "</s>" || tok == "<eos>") {
                eos_id_ = id;
            } else if (tok == "<pad>" || tok == "<|padding|>") {
                pad_id_ = id;
            } else if (tok == "<unk>") {
                unk_id_ = id;
            }
        }
    }

    if (bos_id_ < 0 && eos_id_ >= 0) bos_id_ = eos_id_;
}

namespace {

auto is_letter_utf8(std::uint32_t cp) -> bool {

    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    }
    if (cp <= 0x024F) {
        return true;
    }
    return false;
}
auto is_digit_utf8(std::uint32_t cp) -> bool {
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    return false;
}
auto is_whitespace_utf8(std::uint32_t cp) -> bool {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return true;
    return false;
}

auto decode_utf8(const std::string& s, std::size_t& i) -> std::uint32_t {
    auto b0 = static_cast<std::uint8_t>(s[i]);
    if (b0 < 0x80) { ++i; return b0; }
    if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        auto b1 = static_cast<std::uint8_t>(s[i + 1]);
        std::uint32_t cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
        i += 2; return cp;
    }
    if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        auto b1 = static_cast<std::uint8_t>(s[i + 1]);
        auto b2 = static_cast<std::uint8_t>(s[i + 2]);
        std::uint32_t cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
        i += 3; return cp;
    }
    if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        auto b1 = static_cast<std::uint8_t>(s[i + 1]);
        auto b2 = static_cast<std::uint8_t>(s[i + 2]);
        auto b3 = static_cast<std::uint8_t>(s[i + 3]);
        std::uint32_t cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                            ((b2 & 0x3F) << 6) | (b3 & 0x3F);
        i += 4; return cp;
    }

    ++i; return 0xFFFD;
}

auto encode_utf8(std::uint32_t cp, std::string& out) -> void {
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
}

auto byte_level_pretokenize(const std::string& text,
                            const int (&b2u)[256]) -> std::vector<std::string> {

    struct Piece {
        std::string bytes;
    };
    std::vector<Piece> pieces;

    std::size_t i = 0;
    while (i < text.size()) {

        if (i + 1 < text.size() && text[i] == '\'' ) {
            char c = text[i + 1];
            if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                pieces.push_back({text.substr(i, 2)});
                i += 2;
                continue;
            }
            if (i + 2 < text.size()) {
                if ((c == 'r' && text[i + 2] == 'e') ||
                    (c == 'v' && text[i + 2] == 'e') ||
                    (c == 'l' && text[i + 2] == 'l')) {
                    pieces.push_back({text.substr(i, 3)});
                    i += 3;
                    continue;
                }
            }
        }

        std::size_t start = i;
        std::uint32_t cp = decode_utf8(text, i);
        bool is_ws   = is_whitespace_utf8(cp);
        bool is_let  = is_letter_utf8(cp);
        bool is_dig  = is_digit_utf8(cp);

        if (is_ws && cp == ' ' && i < text.size()) {
            std::size_t after_space = i;
            std::uint32_t cp2 = decode_utf8(text, after_space);
            if (is_letter_utf8(cp2)) {
                std::size_t cur = after_space;
                while (cur < text.size()) {
                    std::size_t probe = cur;
                    std::uint32_t c2 = decode_utf8(text, probe);
                    if (!is_letter_utf8(c2)) break;
                    cur = probe;
                }
                pieces.push_back({text.substr(start, cur - start)});
                i = cur;
                continue;
            }
            if (is_digit_utf8(cp2)) {

                std::size_t cur = after_space;
                std::string lead = text.substr(start, after_space - start);
                while (cur < text.size()) {
                    std::size_t probe = cur;
                    std::uint32_t c2 = decode_utf8(text, probe);
                    if (!is_digit_utf8(c2)) break;
                    pieces.push_back({lead + text.substr(cur, probe - cur)});
                    cur = probe;
                }
                i = cur;
                continue;
            }

        }

        if (is_let) {

            std::size_t cur = i;
            while (cur < text.size()) {
                std::size_t probe = cur;
                std::uint32_t c2 = decode_utf8(text, probe);
                if (!is_letter_utf8(c2)) break;
                cur = probe;
            }
            pieces.push_back({text.substr(start, cur - start)});
            i = cur;
            continue;
        }
        if (is_dig) {

            std::size_t cur = i;
            while (cur < text.size()) {
                std::size_t probe = cur;
                std::uint32_t c2 = decode_utf8(text, probe);
                if (!is_digit_utf8(c2)) break;
                pieces.push_back({text.substr(cur, probe - cur)});
                cur = probe;
            }
            i = cur;
            continue;
        }
        if (!is_ws) {

            std::size_t cur = i;
            while (cur < text.size()) {
                std::size_t probe = cur;
                std::uint32_t c2 = decode_utf8(text, probe);
                if (is_whitespace_utf8(c2) ||
                    is_letter_utf8(c2) ||
                    is_digit_utf8(c2)) break;
                cur = probe;
            }
            pieces.push_back({text.substr(start, cur - start)});
            i = cur;
            continue;
        }

        std::size_t cur = i;
        while (cur < text.size()) {
            std::size_t probe = cur;
            std::uint32_t c2 = decode_utf8(text, probe);
            if (!is_whitespace_utf8(c2)) break;
            cur = probe;
        }
        pieces.push_back({text.substr(start, cur - start)});
        i = cur;
    }

    std::vector<std::string> out;
    out.reserve(pieces.size());
    for (const auto& p : pieces) {
        std::string mapped;
        mapped.reserve(p.bytes.size());
        for (unsigned char byte : p.bytes) {
            int cp = b2u[byte];
            encode_utf8(static_cast<std::uint32_t>(cp), mapped);
        }
        out.push_back(std::move(mapped));
    }
    return out;
}

auto bpe_apply(const std::string&                              word,
               const std::vector<std::pair<std::string,
                                           std::string>>&     merges,
               const std::unordered_map<std::string,
                                         std::int32_t>&      vocab,
               std::int32_t                                    unk_id) -> std::vector<std::string> {

    std::vector<std::string> symbols;
    {
        std::size_t i = 0;
        while (i < word.size()) {
            std::size_t probe = i;
            std::uint32_t cp = decode_utf8(word, probe);
            (void)cp;
            symbols.emplace_back(word.data() + i, probe - i);
            i = probe;
        }
    }
    if (symbols.empty()) return {};

    std::vector<std::pair<std::string, std::string>> pairs;
    pairs.reserve(symbols.size() - 1);
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
        pairs.emplace_back(symbols[i], symbols[i + 1]);
    }

    std::map<std::pair<std::string, std::string>, int> rank;
    // merges 的先后顺序就是 BPE 优先级，数值越小越先合并。
    for (std::size_t i = 0; i < merges.size(); ++i) {
        rank[merges[i]] = static_cast<int>(i);
    }

    while (!pairs.empty()) {

        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_idx = 0;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            auto it = rank.find(pairs[i]);
            if (it == rank.end()) continue;
            if (it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break;

        symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
        pairs.erase(pairs.begin() + static_cast<std::ptrdiff_t>(best_idx));
        if (best_idx > 0) {
            pairs[best_idx - 1] = std::make_pair(symbols[best_idx - 1], symbols[best_idx]);
        }
        if (best_idx < pairs.size()) {
            pairs[best_idx] = std::make_pair(symbols[best_idx], symbols[best_idx + 1]);
        }
    }
    (void)unk_id;
    (void)vocab;
    return symbols;
}

}

auto HuggingFaceBpeTokenizer::encode(const std::string& text) const
    -> std::vector<std::int32_t> {
    if (id_to_token_.empty()) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer::encode: tokenizer not loaded");
    }
    std::vector<std::int32_t> out;
    if (text.empty()) return out;

    auto pretokens = byte_level_pretokenize(text, byte_to_unicode_);
    out.reserve(pretokens.size());
    for (const auto& pre : pretokens) {

        auto it = token_to_id_.find(pre);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            continue;
        }

        auto symbols = bpe_apply(pre, merges_, token_to_id_, unk_id_);
        for (const auto& sym : symbols) {
            auto vit = token_to_id_.find(sym);
            if (vit != token_to_id_.end()) {
                out.push_back(vit->second);
            } else if (unk_id_ >= 0) {
                out.push_back(unk_id_);
            }

        }
    }
    return out;
}

auto HuggingFaceBpeTokenizer::decode(std::span<const std::int32_t> ids) const
    -> std::string {
    if (id_to_token_.empty()) {
        throw std::runtime_error(
            "smollm2::HuggingFaceBpeTokenizer::decode: tokenizer not loaded");
    }
    std::string out;
    out.reserve(ids.size() * 4);
    for (std::int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
            continue;
        }

        if (id == bos_id_ || id == eos_id_ || id == pad_id_) continue;

        const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];

        std::size_t i = 0;
        while (i < tok.size()) {
            std::size_t probe = i;
            std::uint32_t cp = decode_utf8(tok, probe);
            auto b = unicode_to_byte_[cp];
            if (b != kUnmapped) {
                out += static_cast<char>(b);
            } else {

                out += '?';
            }
            i = probe;
        }
    }
    return out;
}

}
