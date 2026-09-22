#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../examples/smollm2/tokenizer.h"
#include "../test_model_assets.h"

using velomind::examples::smollm2::HuggingFaceBpeTokenizer;

namespace {

std::string find_test_tokenizer_json() {
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        velomind_test::get_smollm2_model_dir() / "tokenizer.json",
        velomind_test::get_model_assets_root() / "SmolLM2-135M-Instruct" / "tokenizer.json",
    };
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            return p.string();
        }
    }
    return "";
}

}

TEST_CASE("HuggingFaceBpeTokenizer - uninitialized state",
          "[tokenizer][smollm2]") {
    HuggingFaceBpeTokenizer tok;
    REQUIRE_FALSE(tok.loaded());
    REQUIRE(tok.vocab_size() == 0);
    REQUIRE(tok.bos_id() == -1);
    REQUIRE(tok.eos_id() == -1);

    REQUIRE_THROWS_AS(tok.encode("hello"), std::runtime_error);
    REQUIRE_THROWS_AS(
        tok.decode(std::vector<std::int32_t>{1, 2, 3}),
        std::runtime_error);
}

TEST_CASE("HuggingFaceBpeTokenizer - load nonexistent file",
          "[tokenizer][smollm2]") {
    HuggingFaceBpeTokenizer tok;
    REQUIRE_THROWS_AS(tok.load("/nonexistent/path/to/tokenizer.json"),
                      std::runtime_error);
    REQUIRE_FALSE(tok.loaded());
}

TEST_CASE("HuggingFaceBpeTokenizer - load + round-trip ASCII",
          "[tokenizer][smollm2]") {
    const std::string path = find_test_tokenizer_json();
    if (path.empty()) {
        if (velomind_test::is_model_pipeline_required()) {
            FAIL("模型专用流水线要求资产必须存在: [ASSET_MISSING] SmolLM2 tokenizer.json");
        } else {
            SKIP("跳过分词器测试: [ASSET_MISSING] SmolLM2 tokenizer.json 未找到");
        }
    }

    HuggingFaceBpeTokenizer tok;
    REQUIRE_NOTHROW(tok.load(path));
    REQUIRE(tok.loaded());
    REQUIRE(tok.vocab_size() == 49152);

    REQUIRE(tok.bos_id() == 0);
    REQUIRE(tok.eos_id() == 2);

    SECTION("ASCII round-trip") {

        const std::string original = "Hello world";
        std::vector<std::int32_t> ids;
        REQUIRE_NOTHROW(ids = tok.encode(original));
        REQUIRE_FALSE(ids.empty());

        std::string decoded;
        REQUIRE_NOTHROW(decoded = tok.decode(ids));

        INFO("original=\"" << original << "\" decoded=\"" << decoded << "\"");

        auto trim = [](std::string s) {
            auto a = s.find_first_not_of(' ');
            if (a == std::string::npos) return std::string{};
            auto b = s.find_last_not_of(' ');
            return s.substr(a, b - a + 1);
        };
        REQUIRE(trim(decoded) == original);
    }

    SECTION("Special tokens are skipped during decode") {

        std::vector<std::int32_t> only_special = {tok.bos_id(),
                                                  tok.eos_id(),
                                                  tok.eos_id()};
        std::string decoded;
        REQUIRE_NOTHROW(decoded = tok.decode(only_special));
        REQUIRE(decoded.empty());
    }
}

TEST_CASE("HuggingFaceBpeTokenizer - encode simple text returns sensible ids",
          "[tokenizer][smollm2]") {
    const std::string path = find_test_tokenizer_json();
    if (path.empty()) {
        if (velomind_test::is_model_pipeline_required()) {
            FAIL("模型专用流水线要求资产必须存在: [ASSET_MISSING] SmolLM2 tokenizer.json");
        } else {
            SKIP("跳过分词器测试: [ASSET_MISSING] SmolLM2 tokenizer.json 未找到");
        }
    }

    HuggingFaceBpeTokenizer tok;
    REQUIRE_NOTHROW(tok.load(path));

    const std::vector<std::string> cases = {"Hello", "The quick brown fox",
                                              "1 2 3 4 5", "..."};
    for (const auto& s : cases) {
        auto ids = tok.encode(s);
        for (auto id : ids) {
            REQUIRE(id >= 0);
            REQUIRE(id < tok.vocab_size());
        }
    }
}
