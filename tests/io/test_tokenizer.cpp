#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "examples/tinyllama/tokenizer.h"

namespace {

std::string find_test_model() {
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        "examples/tinyllama/testdata/test.model",
        "../examples/tinyllama/testdata/test.model",
        "../../examples/tinyllama/testdata/test.model",
        "/home/aiza/workspace/dev/velo-mind/examples/tinyllama/testdata/test.model",
    };
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            return p.string();
        }
    }
    return "";
}

}

TEST_CASE("Tokenizer - uninitialized state", "[tokenizer][tinyllama]") {
    velomind::examples::tinyllama::Tokenizer tok;
    REQUIRE_FALSE(tok.loaded());
    REQUIRE(tok.vocab_size() == 0);
    REQUIRE(tok.bos_id() == -1);
    REQUIRE(tok.eos_id() == -1);

    REQUIRE_THROWS_AS(tok.encode("hello"), std::runtime_error);
    REQUIRE_THROWS_AS(
        tok.decode(std::vector<std::int32_t>{1, 2, 3}),
        std::runtime_error);
}

TEST_CASE("Tokenizer - load nonexistent file", "[tokenizer][tinyllama]") {
    velomind::examples::tinyllama::Tokenizer tok;
    REQUIRE_THROWS_AS(tok.load("/nonexistent/path/to/model.model"), std::runtime_error);
    REQUIRE_FALSE(tok.loaded());
}

TEST_CASE("Tokenizer - load, encode, and decode round-trip", "[tokenizer][tinyllama]") {
    const std::string model_path = find_test_model();
    if (model_path.empty()) {
        WARN("test.model not found, skipping live encode/decode test");
        return;
    }

    velomind::examples::tinyllama::Tokenizer tok;
    REQUIRE_NOTHROW(tok.load(model_path));
    REQUIRE(tok.loaded());
    REQUIRE(tok.vocab_size() == 1000);
    REQUIRE(tok.bos_id() == 1);
    REQUIRE(tok.eos_id() == 2);

    const std::string original = "Hello world! Testing sentencepiece integration.";
    std::vector<std::int32_t> tokens;
    REQUIRE_NOTHROW(tokens = tok.encode(original));
    REQUIRE_FALSE(tokens.empty());

    std::string decoded;
    REQUIRE_NOTHROW(decoded = tok.decode(tokens));
    REQUIRE(decoded == original);

    velomind::examples::tinyllama::Tokenizer tok_moved(std::move(tok));
    REQUIRE(tok_moved.loaded());
    REQUIRE(tok_moved.vocab_size() == 1000);
    REQUIRE(tok_moved.decode(tokens) == original);
}
