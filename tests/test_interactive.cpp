#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "llama_engine.h"
#include "sample.h"
#include "tokenizer.h"
#include "tinyllama_engine.h"
#include "velomind/device.h"
#include "velomind/types.h"

using namespace velomind;
using namespace velomind::examples::tinyllama;
using velomind::examples::llama::TokenizerAdapter;
using velomind::examples::llama::LlamaTextStreamer;

namespace {

std::string find_test_model() {
    namespace fs = std::filesystem;
    const std::vector<fs::path> candidates = {
        "examples/tinyllama/testdata/test.model",
        "../examples/tinyllama/testdata/test.model",
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

TEST_CASE("TextStreamer - incremental token-to-text streaming",
          "[interactive][tinyllama]") {
    Tokenizer tok;
    const std::string sp_path = find_test_model();
    if (sp_path.empty()) {
        WARN("test.model not found, skipping live TextStreamer test");
        return;
    }
    tok.load(sp_path);
    REQUIRE(tok.loaded());

    const std::string full_prompt = "Hello world! Testing sentencepiece.";
    std::vector<std::int32_t> tokens = tok.encode(full_prompt);
    REQUIRE(tokens.size() >= 3);

    SECTION("Streaming without initial prompt prefix") {
        std::vector<std::string> streamed_pieces;
        LlamaTextStreamer streamer(tok, [&](const std::string& piece) {
            streamed_pieces.push_back(piece);
        });

        std::string accumulated;
        for (std::int32_t tok_id : tokens) {
            std::string piece = streamer.put(tok_id);
            accumulated += piece;
        }
        std::string remaining = streamer.flush();
        accumulated += remaining;

        REQUIRE(streamer.generated_tokens().size() == tokens.size());
        REQUIRE(streamer.text() == accumulated);
        REQUIRE(accumulated == full_prompt);
    }

    SECTION("Streaming with initial prompt prefix (context continuation)") {

        std::vector<std::int32_t> prompt_tokens(tokens.begin(),
                                                tokens.begin() + 2);
        std::vector<std::int32_t> gen_tokens(tokens.begin() + 2, tokens.end());

        std::string streamed_generated;
        LlamaTextStreamer streamer(tok, [&](const std::string& piece) {
            streamed_generated += piece;
        });
        streamer.reset(prompt_tokens);

        for (std::int32_t tok_id : gen_tokens) {
            streamer.put(tok_id);
        }
        streamer.flush();

        std::vector<std::int32_t> all_tokens(prompt_tokens.begin(),
                                              prompt_tokens.end());
        all_tokens.insert(all_tokens.end(), gen_tokens.begin(), gen_tokens.end());
        std::string expected_full = tok.decode(all_tokens);
        std::vector<std::int32_t> just_prompt(prompt_tokens.begin(),
                                              prompt_tokens.end());
        std::string prompt_text = tok.decode(just_prompt);

        REQUIRE(streamer.generated_tokens().size() == gen_tokens.size());
        REQUIRE(prompt_text + streamer.text() == expected_full);
    }

    SECTION("Streamer handles unloaded tokenizer gracefully") {
        Tokenizer empty_tok;
        LlamaTextStreamer streamer(empty_tok);
        std::string p1 = streamer.put(42);
        std::string p2 = streamer.put(100);
        REQUIRE(p1 == "[42]");
        REQUIRE(p2 == "[100]");
        REQUIRE(streamer.text() == "[42][100]");
    }
}

TEST_CASE("TinyLlamaEngine - synthetic mode weight sharing and multi-step inference",
          "[interactive][tinyllama]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CPU;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_FALSE(engine.is_loaded());

    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    SECTION("Consecutive forward passes with varying sequence lengths (weight sharing)") {

        std::vector<std::int32_t> tok2 = {1, 2};
        auto logits2 = engine.forward_logits(tok2);
        REQUIRE(logits2.size() == engine.config().model_config.vocab_size);
        for (float v : logits2) {
            REQUIRE(std::isfinite(v));
        }

        std::vector<std::int32_t> tok4 = {1, 2, 3, 4};
        auto logits4 = engine.forward_logits(tok4);
        REQUIRE(logits4.size() == engine.config().model_config.vocab_size);
        for (float v : logits4) {
            REQUIRE(std::isfinite(v));
        }

        std::vector<std::int32_t> empty;
        REQUIRE_THROWS_AS(engine.forward_logits(empty), std::invalid_argument);
    }

    SECTION("End-to-end generate with streaming callback") {
        SamplerConfig scfg{.temperature = 0.0f};
        const std::size_t max_new = 4;
        std::mt19937 rng(1234);
        auto sampler_fn = make_sampler_fn(scfg, rng);

        std::vector<std::int32_t> stream_tokens;
        std::string stream_text;

        auto res = engine.generate(
            std::vector<std::int32_t>{1, 2},
            sampler_fn,
            max_new,
            1234,
            [&](std::int32_t tok_id, const std::string& piece) {
                if (tok_id >= 0) {
                    stream_tokens.push_back(tok_id);
                }
                stream_text += piece;
            }
        );

        REQUIRE(res.generated_tokens.size() > 0);
        REQUIRE(res.generated_tokens.size() <= max_new);
        REQUIRE(stream_tokens == res.generated_tokens);
        REQUIRE(res.text == stream_text);
        REQUIRE(res.elapsed_seconds >= 0.0);
    }
}

#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("TinyLlamaEngine - ISPC backend multi-step generation",
          "[interactive][tinyllama][ispc]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::ISPC;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    SamplerConfig scfg{.temperature = 0.0f};
    std::mt19937 rng(42);
    auto sampler_fn = make_sampler_fn(scfg, rng);

    auto res = engine.generate(
        std::vector<std::int32_t>{0, 1},
        sampler_fn,
        3,
        42,
        [](std::int32_t, const std::string&) {}
    );

    REQUIRE(res.generated_tokens.size() == 3);
    for (std::int32_t t : res.generated_tokens) {
        REQUIRE(t >= 0);
        REQUIRE(t < static_cast<std::int32_t>(engine.config().model_config.vocab_size));
    }
}
#endif

TEST_CASE("TinyLlamaEngine - KV Cache session reset and step tracking",
          "[interactive][kv_cache][tinyllama]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CPU;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    // 初始状态下会话应为空
    engine.reset_session();
    REQUIRE(engine.session_seq_len() == 0);
    REQUIRE(engine.inner().kv_cache().empty());

    // Prefill 4 个 token，建立各层 KV cache（kTiny 词表大小为 16）
    std::vector<std::int32_t> prompt = {1, 2, 3, 4};
    auto logits_prefill = engine.prefill(prompt);
    REQUIRE(logits_prefill.size() == engine.config().model_config.vocab_size);
    REQUIRE(engine.session_seq_len() == 4);
    REQUIRE_FALSE(engine.inner().kv_cache().empty());
    REQUIRE(engine.inner().kv_cache().layers.size() == engine.config().model_config.num_layers);

    // 单 token decode 步骤 1
    auto logits_step1 = engine.forward_step(5, engine.session_seq_len());
    REQUIRE(logits_step1.size() == engine.config().model_config.vocab_size);
    REQUIRE(engine.session_seq_len() == 5);

    // 单 token decode 步骤 2
    auto logits_step2 = engine.forward_step(6, engine.session_seq_len());
    REQUIRE(logits_step2.size() == engine.config().model_config.vocab_size);
    REQUIRE(engine.session_seq_len() == 6);

    // 会话重置：清除全部逐层 cache，序列长度归零
    engine.reset_session();
    REQUIRE(engine.session_seq_len() == 0);
    REQUIRE(engine.inner().kv_cache().empty());
}

TEST_CASE("TinyLlamaEngine - KV Cache incremental decode vs full prefill reference alignment",
          "[interactive][kv_cache][alignment]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CPU;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    // 8 个 prompt token
    std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::int32_t> accumulated = prompt;

    // 路径 A：增量 prefill 与逐 token decode
    engine.reset_session();
    auto prefill_logits = engine.prefill(prompt);

    // 路径 B：无状态全序列参考 prefill
    auto ref_prefill_logits = engine.forward_logits(prompt);

    REQUIRE(prefill_logits.size() == ref_prefill_logits.size());
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < prefill_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(prefill_logits[i] - ref_prefill_logits[i]));
    }
    REQUIRE(max_diff < 1e-4f);

    auto argmax_fn = [](std::span<const float> l) -> std::int32_t {
        return static_cast<std::int32_t>(
            std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    };

    std::int32_t next_token = argmax_fn(prefill_logits);
    std::int32_t ref_next = argmax_fn(ref_prefill_logits);
    REQUIRE(next_token == ref_next);

    // 接下来长上下文自回归逐步前向 16 个 token，验证 logits 误差与贪婪 token 对齐
    const std::size_t decode_steps = 16;
    for (std::size_t step = 0; step < decode_steps; ++step) {
        accumulated.push_back(next_token);

        auto decode_logits = engine.forward_step(next_token, engine.session_seq_len());
        auto ref_step_logits = engine.forward_logits(accumulated);

        REQUIRE(decode_logits.size() == ref_step_logits.size());
        float step_max_diff = 0.0f;
        for (std::size_t i = 0; i < decode_logits.size(); ++i) {
            step_max_diff = std::max(step_max_diff, std::abs(decode_logits[i] - ref_step_logits[i]));
        }
        REQUIRE(step_max_diff < 1e-4f);

        std::int32_t cand = argmax_fn(decode_logits);
        std::int32_t cand_ref = argmax_fn(ref_step_logits);
        REQUIRE(cand == cand_ref);

        next_token = cand;
    }
}

TEST_CASE("TinyLlamaEngine - KV Cache micro-benchmark and performance comparison",
          "[interactive][kv_cache][benchmark]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CPU;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    const std::size_t prompt_len = 8;
    const std::size_t decode_tokens = 16;
    std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};

    // Prefill 耗时与内存
    auto t_prefill_start = std::chrono::steady_clock::now();
    engine.reset_session();
    auto prefill_logits = engine.prefill(prompt);
    auto t_prefill_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    REQUIRE(prefill_logits.size() == engine.config().model_config.vocab_size);

    auto argmax_fn = [](std::span<const float> l) -> std::int32_t {
        return static_cast<std::int32_t>(
            std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    };
    std::int32_t tok = argmax_fn(prefill_logits);

    // 带 KV cache 增量单 token decode 耗时
    auto t_kv_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < decode_tokens; ++i) {
        auto step_logits = engine.forward_step(tok, engine.session_seq_len());
        tok = argmax_fn(step_logits);
    }
    auto t_kv_end = std::chrono::steady_clock::now();
    double kv_total_ms = std::chrono::duration<double, std::milli>(t_kv_end - t_kv_start).count();
    double kv_per_tok_ms = kv_total_ms / static_cast<double>(decode_tokens);
    double kv_tok_per_sec = (decode_tokens * 1000.0) / kv_total_ms;

    // 无 KV cache 全序列重复 prefill 耗时
    std::vector<std::int32_t> full_seq = prompt;
    tok = argmax_fn(prefill_logits);
    auto t_full_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < decode_tokens; ++i) {
        full_seq.push_back(tok);
        auto step_logits = engine.forward_logits(full_seq);
        tok = argmax_fn(step_logits);
    }
    auto t_full_end = std::chrono::steady_clock::now();
    double full_total_ms = std::chrono::duration<double, std::milli>(t_full_end - t_full_start).count();
    double full_per_tok_ms = full_total_ms / static_cast<double>(decode_tokens);
    double full_tok_per_sec = (decode_tokens * 1000.0) / full_total_ms;

    double speedup = full_total_ms / kv_total_ms;

    std::cout << "\n=== LlamaEngine KV Cache Benchmark (TinyLlama S=" << prompt_len << " -> " << prompt_len + decode_tokens << ") ===\n"
              << "  Prefill Latency:       " << prefill_ms << " ms\n"
              << "  KV Cache Decode:       " << kv_per_tok_ms << " ms/tok (" << kv_tok_per_sec << " tok/s)\n"
              << "  Full Prefill Decode:   " << full_per_tok_ms << " ms/tok (" << full_tok_per_sec << " tok/s)\n"
              << "  Decode Speedup:        " << speedup << "x\n"
              << "========================================================================\n" << std::endl;

    REQUIRE(kv_total_ms > 0.0);
    REQUIRE(full_total_ms > 0.0);
}

TEST_CASE("TinyLlamaEngine - CUDA backend multi-step KV cache generation and alignment",
          "[interactive][kv_cache][cuda]") {
    if (!Device::cuda().is_available()) {
        SKIP("CUDA device is not available on this host.");
    }

    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CUDA;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    std::vector<std::int32_t> prompt = {1, 2, 3, 4};
    std::vector<std::int32_t> accumulated = prompt;

    engine.reset_session();
    auto prefill_logits = engine.prefill(prompt);
    auto ref_prefill_logits = engine.forward_logits(prompt);

    REQUIRE(prefill_logits.size() == ref_prefill_logits.size());
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < prefill_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(prefill_logits[i] - ref_prefill_logits[i]));
    }
    REQUIRE(max_diff < 1e-3f);

    auto argmax_fn = [](std::span<const float> l) -> std::int32_t {
        return static_cast<std::int32_t>(
            std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    };
    std::int32_t next_tok = argmax_fn(prefill_logits);

    for (std::size_t step = 0; step < 8; ++step) {
        accumulated.push_back(next_tok);
        auto decode_logits = engine.forward_step(next_tok, engine.session_seq_len());
        auto ref_logits = engine.forward_logits(accumulated);

        float step_diff = 0.0f;
        for (std::size_t i = 0; i < decode_logits.size(); ++i) {
            step_diff = std::max(step_diff, std::abs(decode_logits[i] - ref_logits[i]));
        }
        REQUIRE(step_diff < 1e-3f);
        std::int32_t cand = argmax_fn(decode_logits);
        std::int32_t cand_ref = argmax_fn(ref_logits);
        REQUIRE(cand == cand_ref);
        next_tok = cand;
    }
}

TEST_CASE("TinyLlamaEngine - last token logits path optimization and forward_full_logits equivalence",
          "[interactive][logits][last_token]") {
    EngineConfig cfg;
    cfg.use_synthetic = true;
    cfg.device = DeviceType::CPU;
    cfg.tokenizer_path = find_test_model();

    TinyLlamaEngine engine(cfg);
    REQUIRE_NOTHROW(engine.load());
    REQUIRE(engine.is_loaded());

    const std::size_t vocab = engine.config().model_config.vocab_size;

    // 单 token prompt (S = 1)
    std::vector<std::int32_t> single_tok = {7};
    auto single_prefill = engine.prefill(single_tok);
    auto single_full    = engine.forward_full_logits(single_tok);
    auto single_last    = engine.forward_logits(single_tok);

    REQUIRE(single_prefill.size() == vocab);
    REQUIRE(single_full.size() == vocab);
    REQUIRE(single_last.size() == vocab);
    REQUIRE(single_prefill == single_full);
    REQUIRE(single_prefill == single_last);

    // 多 token 序列 (S = 8)
    std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::size_t S = prompt.size();

    engine.reset_session();
    auto prefill_logits = engine.prefill(prompt);
    auto full_logits    = engine.forward_full_logits(prompt);
    auto last_logits    = engine.forward_logits(prompt);

    // 验证形状与容量契约：生成路径仅返回 vocab，全序列返回 S * vocab
    REQUIRE(prefill_logits.size() == vocab);
    REQUIRE(full_logits.size() == S * vocab);
    REQUIRE(last_logits.size() == vocab);

    // 验证全序列最后一行切片与优化路径 prefill 精确一致
    float max_diff_prefill_vs_full = 0.0f;
    for (std::size_t v = 0; v < vocab; ++v) {
        float full_val = full_logits[(S - 1) * vocab + v];
        max_diff_prefill_vs_full = std::max(
            max_diff_prefill_vs_full, std::abs(prefill_logits[v] - full_val));
    }
    REQUIRE(max_diff_prefill_vs_full < 1e-4f);

    // 验证局部 D2H 读取的 last_logits 与 prefill_logits 精确一致
    float max_diff_last_vs_prefill = 0.0f;
    for (std::size_t v = 0; v < vocab; ++v) {
        max_diff_last_vs_prefill = std::max(
            max_diff_last_vs_prefill, std::abs(last_logits[v] - prefill_logits[v]));
    }
    REQUIRE(max_diff_last_vs_prefill < 1e-4f);
}
