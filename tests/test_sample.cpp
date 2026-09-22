#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "examples/tinyllama/sample.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

using namespace velomind;
using namespace velomind::examples::tinyllama;

TEST_CASE("sample_argmax - greedy selection and edge cases", "[sampling][tinyllama]") {
    SECTION("Normal distinct logits") {
        const std::vector<float> logits = {0.1f, -0.5f, 2.5f, 1.2f, -3.0f};
        REQUIRE(sample_argmax(logits) == 2);
    }

    SECTION("All negative logits") {
        const std::vector<float> logits = {-10.5f, -2.1f, -5.0f, -8.3f};
        REQUIRE(sample_argmax(logits) == 1);
    }

    SECTION("Single element") {
        const std::vector<float> logits = {42.0f};
        REQUIRE(sample_argmax(logits) == 0);
    }

    SECTION("Flat logits (ties return first maximal index)") {
        const std::vector<float> logits = {3.0f, 3.0f, 1.0f};
        REQUIRE(sample_argmax(logits) == 0);
    }

    SECTION("Handles NaN by skipping them") {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> logits = {qnan, 1.0f, qnan, 5.0f, 2.0f};
        REQUIRE(sample_argmax(logits) == 3);
    }

    SECTION("Throws on empty logits") {
        const std::vector<float> empty;
        REQUIRE_THROWS_AS(sample_argmax(empty), std::invalid_argument);
    }

    SECTION("Throws when all logits are NaN") {
        const float qnan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> all_nan = {qnan, qnan, qnan};
        REQUIRE_THROWS_AS(sample_argmax(all_nan), std::runtime_error);
    }
}

TEST_CASE("sample_top_p - nucleus sampling behavior", "[sampling][tinyllama]") {
    std::mt19937 rng(1337);

    SECTION("Falls back to argmax when temperature <= 0") {
        const std::vector<float> logits = {1.0f, 10.0f, 2.0f};
        REQUIRE(sample_top_p(logits, 0.0f, 0.9f, rng) == 1);
        REQUIRE(sample_top_p(logits, -1.0f, 0.9f, rng) == 1);
    }

    SECTION("Falls back to argmax when top_p <= 0") {
        const std::vector<float> logits = {1.0f, 10.0f, 2.0f};
        REQUIRE(sample_top_p(logits, 0.8f, 0.0f, rng) == 1);
        REQUIRE(sample_top_p(logits, 0.8f, -0.5f, rng) == 1);
    }

    SECTION("Dominant logit is sampled with near 100% probability") {
        const std::vector<float> logits = {100.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 50; ++i) {
            REQUIRE(sample_top_p(logits, 1.0f, 0.9f, rng) == 0);
        }
    }

    SECTION("Cutoff strictly excludes tail tokens") {

        const std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
        std::map<std::int32_t, int> counts;
        for (int i = 0; i < 500; ++i) {
            std::int32_t tok = sample_top_p(logits, 1.0f, 0.5f, rng);
            counts[tok]++;
        }
        REQUIRE(counts[0] > 0);
        REQUIRE(counts[1] > 0);
        REQUIRE(counts[2] == 0);
        REQUIRE(counts[3] == 0);
    }

    SECTION("Sampling is deterministic with identical RNG seed") {
        const std::vector<float> logits = {0.5f, 1.2f, -0.3f, 2.0f, 0.8f};
        std::mt19937 rng1(42);
        std::mt19937 rng2(42);

        for (int i = 0; i < 20; ++i) {
            REQUIRE(sample_top_p(logits, 0.7f, 0.85f, rng1) ==
                    sample_top_p(logits, 0.7f, 0.85f, rng2));
        }
    }

    SECTION("Throws on empty logits") {
        const std::vector<float> empty;
        REQUIRE_THROWS_AS(sample_top_p(empty, 1.0f, 0.9f, rng), std::invalid_argument);
    }
}

TEST_CASE("sample_token - config dispatch", "[sampling][tinyllama]") {
    std::mt19937 rng(42);
    const std::vector<float> logits = {1.0f, 20.0f, 3.0f};

    SamplerConfig greedy_cfg{.temperature = 0.0f, .top_p = 0.9f};
    REQUIRE(sample_token(logits, greedy_cfg, rng) == 1);

    SamplerConfig stoch_cfg{.temperature = 1.0f, .top_p = 0.9f};
    REQUIRE(sample_token(logits, stoch_cfg, rng) == 1);
}

TEST_CASE("read_last_token_logits - Tensor extraction", "[sampling][tinyllama]") {
    Graph g;

    SECTION("1-D Tensor [vocab]") {
        auto t = g.input({4}, DataType::Float32);
        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        const std::vector<float> vals = {1.5f, -2.5f, 3.0f, 0.0f};
        const auto bytes = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(vals.data()), vals.size() * sizeof(float));
        t.copy_from_host(bytes);

        auto extracted = read_last_token_logits(t);
        REQUIRE(extracted.size() == 4);
        for (std::size_t i = 0; i < 4; ++i) {
            REQUIRE(extracted[i] == Catch::Approx(vals[i]));
        }

        std::mt19937 rng(42);
        SamplerConfig cfg{.temperature = 0.0f};
        REQUIRE(sample_next_token(t, cfg, rng) == 2);
    }

    SECTION("2-D Tensor [seq, vocab]") {
        auto t = g.input({3, 4}, DataType::Float32);
        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        const std::vector<float> vals = {

            0.1f, 0.2f, 0.3f, 0.4f,

            1.0f, 2.0f, 3.0f, 4.0f,

            -5.0f, 10.0f, 0.0f, 1.0f
        };
        const auto bytes = std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(vals.data()), vals.size() * sizeof(float));
        t.copy_from_host(bytes);

        auto extracted = read_last_token_logits(t);
        REQUIRE(extracted.size() == 4);
        REQUIRE(extracted[0] == Catch::Approx(-5.0f));
        REQUIRE(extracted[1] == Catch::Approx(10.0f));
        REQUIRE(extracted[2] == Catch::Approx(0.0f));
        REQUIRE(extracted[3] == Catch::Approx(1.0f));

        std::mt19937 rng(42);
        SamplerConfig cfg{.temperature = 0.0f};
        REQUIRE(sample_next_token(t, cfg, rng) == 1);
    }

    SECTION("Invalid handle or shape throws") {
        Tensor null_tensor;
        REQUIRE_THROWS_AS(read_last_token_logits(null_tensor), std::runtime_error);

        auto t_3d = g.input({2, 3, 4}, DataType::Float32);
        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);
        REQUIRE_THROWS_AS(read_last_token_logits(t_3d), std::runtime_error);
    }
}

TEST_CASE("generate_tokens - autoregressive generation loop", "[sampling][tinyllama]") {
    std::mt19937 rng(42);

    SECTION("Generates expected number of tokens") {

        auto mock_forward = [](std::span<const std::int32_t> current) -> std::vector<float> {
            std::vector<float> logits(10, 0.0f);
            std::int32_t next_id = current.empty() ? 0 : (current.back() + 1) % 10;
            logits[next_id] = 50.0f;
            return logits;
        };

        const std::vector<std::int32_t> prompt = {3};
        SamplerConfig cfg{.temperature = 0.0f};

        std::vector<std::int32_t> notified_tokens;
        auto gen = generate_tokens(
            prompt, 4, cfg, -1, rng, mock_forward,
            [&](std::int32_t tok) { notified_tokens.push_back(tok); });

        const std::vector<std::int32_t> expected = {4, 5, 6, 7};
        REQUIRE(gen == expected);
        REQUIRE(notified_tokens == expected);
    }

    SECTION("Early stop on EOS token") {
        const std::int32_t kEos = 9;
        auto mock_forward = [kEos](std::span<const std::int32_t> current) -> std::vector<float> {
            std::vector<float> logits(10, 0.0f);
            if (current.size() >= 3) {
                logits[kEos] = 50.0f;
            } else {
                logits[1] = 50.0f;
            }
            return logits;
        };

        const std::vector<std::int32_t> prompt = {0};
        SamplerConfig cfg{.temperature = 0.0f};

        auto gen = generate_tokens(prompt, 10, cfg, kEos, rng, mock_forward);

        REQUIRE(gen.size() == 3);
        REQUIRE(gen.back() == kEos);
    }
}
