#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/safetensors.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "graph_builder.h"
#include "model.h"
#include "model_loader.h"
#include "sample.h"

#include "llama_rope_mask.h"

#include "test_model_assets.h"

using namespace velomind;
using namespace velomind::examples::tinyllama;
using velomind::examples::llama::compute_rope_cache;
using velomind::examples::llama::compute_causal_mask;

static void run_tinyllama_fixed_asset(DeviceType device) {
    if (!Device(device).is_available()) {
        SKIP(std::string("设备不可用: ") + Device(device).name());
    }
    const auto small_model_path = velomind_test::require_or_skip_asset(
        "tests/testdata/toy_llama/model.safetensors", "TinyLlama toy fixed asset");

    auto model_sf = load_safetensors(small_model_path);
    REQUIRE(model_sf.has_value());

    const Config cfg = {
        .vocab_size        = 256,
        .hidden_size       = 64,
        .intermediate_size = 128,
        .num_layers        = 2,
        .num_heads         = 4,
        .num_kv_heads      = 4,
        .rms_norm_eps      = 1e-5f,
        .rope_theta        = 10000.0f,
        .max_seq_len       = 512,
    };
    const std::size_t seq = 5;

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_tinyllama_weights(g, cfg, seq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(device);
    REQUIRE(exec);

    populate_tinyllama_weights(*model_sf, cfg, weights, seq);

    std::vector<int32_t> tok_data = {1, 2, 3, 4, 5};
    tokens.copy_from_host(std::as_bytes(std::span(tok_data)));

    exec->execute();

    std::vector<float> actual_logits(seq * cfg.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(actual_logits)));

    velomind_test::require_all_finite(actual_logits, "actual_logits");

    std::vector<float> last_token_logits(cfg.vocab_size);
    for (std::size_t v = 0; v < cfg.vocab_size; ++v) {
        last_token_logits[v] = actual_logits[(seq - 1) * cfg.vocab_size + v];
    }
    int32_t predicted = sample_argmax(last_token_logits);
    CHECK(predicted >= 0);
    CHECK(predicted < static_cast<int32_t>(cfg.vocab_size));
}

TEST_CASE("TinyLlama - 5-token prefill on fixed small asset (CPU)", "[llm][asset][cpu]") {
    run_tinyllama_fixed_asset(DeviceType::CPU);
}
#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("TinyLlama - 5-token prefill on fixed small asset (ISPC)", "[llm][asset][ispc]") {
    run_tinyllama_fixed_asset(DeviceType::ISPC);
}
#endif
#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("TinyLlama - 5-token prefill on fixed small asset (CUDA)", "[llm][asset][cuda]") {
    run_tinyllama_fixed_asset(DeviceType::CUDA);
}
#endif
#ifdef VELOMIND_ENABLE_VULKAN
TEST_CASE("TinyLlama - 5-token prefill on fixed small asset (Vulkan)", "[llm][asset][vulkan]") {
    run_tinyllama_fixed_asset(DeviceType::VULKAN);
}
#endif

TEST_CASE("TinyLlama - 5-token prefill vs HuggingFace reference", "[.][llm][slow]") {
    const auto ref_path = velomind_test::require_or_skip_asset(
        "examples/tinyllama/testdata/reference_prompt_5tokens.safetensors", "TinyLlama HF reference data");

    auto ref_file = load_safetensors(ref_path);
    REQUIRE(ref_file.has_value());

    const auto& ref_tensors = ref_file->tensors;
    REQUIRE(ref_tensors.count("input_tokens"));
    REQUIRE(ref_tensors.count("reference_logits"));

    const std::size_t seq = 5;
    const auto model_path = velomind_test::get_tinyllama_model_path();
    velomind_test::require_or_skip_asset(model_path, "TinyLlama full model weights");

    auto model_sf = load_safetensors(model_path);
    REQUIRE(model_sf.has_value());

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_tinyllama_weights(g, kReal, seq);

    auto logits = build_forward_graph_prefill(g, kReal, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec);

    populate_tinyllama_weights(*model_sf, kReal, weights, seq);

    const auto& tok_storage = ref_tensors.at("input_tokens");
    tokens.copy_from_host(std::span<const std::byte>(
        static_cast<const std::byte*>(tok_storage->data), tok_storage->size_bytes));

    exec->execute();

    std::vector<float> actual_logits(seq * kReal.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(actual_logits)));

    const auto& ref_logits_storage = ref_tensors.at("reference_logits");
    const float* ref_data = static_cast<const float*>(ref_logits_storage->data);

    velomind_test::require_all_finite(actual_logits, "actual_logits");
    velomind_test::require_all_finite(ref_data, actual_logits.size(), "reference_logits");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < actual_logits.size(); ++i) {
        float diff = std::abs(actual_logits[i] - ref_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    CHECK(max_diff < 1e-3f);

    std::vector<float> last_token_logits(kReal.vocab_size);
    for (std::size_t v = 0; v < kReal.vocab_size; ++v) {
        last_token_logits[v] = actual_logits[(seq - 1) * kReal.vocab_size + v];
    }
    int32_t predicted_token = sample_argmax(last_token_logits);
    CHECK(predicted_token == 29889);
}

#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("TinyLlama - 5-token prefill synthetic structure (ISPC)", "[llm][ispc]") {
    Config cfg = kTiny;
    const std::size_t seq = 5;

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_tinyllama_weights(g, cfg, seq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(DeviceType::ISPC);
    REQUIRE(exec);

    std::vector<int32_t> tok_data = {1, 2, 3, 4, 5};
    tokens.copy_from_host(std::as_bytes(std::span(tok_data)));

    std::vector<float> ones_embed(cfg.vocab_size * cfg.hidden_size, 0.1f);
    weights.w_embed.copy_from_host(std::as_bytes(std::span(ones_embed)));

    std::vector<float> ones_lm(cfg.hidden_size * cfg.vocab_size, 0.1f);
    weights.w_lm_head.copy_from_host(std::as_bytes(std::span(ones_lm)));

    std::vector<float> ones_norm(cfg.hidden_size, 1.0f);
    weights.w_final_norm.copy_from_host(std::as_bytes(std::span(ones_norm)));

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        weights.w_attn_norm[l].copy_from_host(std::as_bytes(std::span(ones_norm)));
        weights.w_ffn_norm[l].copy_from_host(std::as_bytes(std::span(ones_norm)));

        std::vector<float> w_mat(cfg.hidden_size * cfg.hidden_size, 0.05f);
        weights.wq[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wk[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wv[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wo[l].copy_from_host(std::as_bytes(std::span(w_mat)));

        std::vector<float> w_gate_up(cfg.hidden_size * cfg.intermediate_size, 0.05f);
        weights.w_gate[l].copy_from_host(std::as_bytes(std::span(w_gate_up)));
        weights.w_up[l].copy_from_host(std::as_bytes(std::span(w_gate_up)));

        std::vector<float> w_down(cfg.intermediate_size * cfg.hidden_size, 0.05f);
        weights.w_down[l].copy_from_host(std::as_bytes(std::span(w_down)));
    }

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(seq, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf);
    weights.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    weights.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    std::vector<float> mask_buf;
    compute_causal_mask(cfg.num_heads, seq, mask_buf);
    weights.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

    exec->execute();

    std::vector<float> out(seq * cfg.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(out)));

    for (float v : out) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("TinyLlama - 5-token prefill vs HuggingFace reference (ISPC)", "[.][llm][slow][ispc]") {
    const auto ref_path = velomind_test::require_or_skip_asset(
        "examples/tinyllama/testdata/reference_prompt_5tokens.safetensors", "TinyLlama HF reference data");

    auto ref_file = load_safetensors(ref_path);
    REQUIRE(ref_file.has_value());

    const auto& ref_tensors = ref_file->tensors;
    REQUIRE(ref_tensors.count("input_tokens"));
    REQUIRE(ref_tensors.count("reference_logits"));

    const std::size_t seq = 5;
    const auto model_path = velomind_test::get_tinyllama_model_path();
    velomind_test::require_or_skip_asset(model_path, "TinyLlama full model weights");

    auto model_sf = load_safetensors(model_path);
    REQUIRE(model_sf.has_value());

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_tinyllama_weights(g, kReal, seq);

    auto logits = build_forward_graph_prefill(g, kReal, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(DeviceType::ISPC);
    REQUIRE(exec);

    populate_tinyllama_weights(*model_sf, kReal, weights, seq);

    const auto& tok_storage = ref_tensors.at("input_tokens");
    tokens.copy_from_host(std::span<const std::byte>(
        static_cast<const std::byte*>(tok_storage->data), tok_storage->size_bytes));

    exec->execute();

    std::vector<float> actual_logits(seq * kReal.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(actual_logits)));

    const auto& ref_logits_storage = ref_tensors.at("reference_logits");
    const float* ref_data = static_cast<const float*>(ref_logits_storage->data);

    velomind_test::require_all_finite(actual_logits, "actual_logits");
    velomind_test::require_all_finite(ref_data, actual_logits.size(), "reference_logits");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < actual_logits.size(); ++i) {
        float diff = std::abs(actual_logits[i] - ref_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    CHECK(max_diff < 1e-3f);

    std::vector<float> last_token_logits(kReal.vocab_size);
    for (std::size_t v = 0; v < kReal.vocab_size; ++v) {
        last_token_logits[v] = actual_logits[(seq - 1) * kReal.vocab_size + v];
    }
    int32_t predicted_token = sample_argmax(last_token_logits);
    CHECK(predicted_token == 29889);
}
#endif

#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("TinyLlama - 5-token prefill synthetic structure (CUDA)", "[llm][cuda]") {
    if (!Device::cuda().is_available()) {
        SKIP("CUDA device is not available on this host.");
    }
    Config cfg = kTiny;
    const std::size_t seq = 5;

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_tinyllama_weights(g, cfg, seq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    REQUIRE(logits);

    std::unique_ptr<Executable> exec;
    try {
        exec = g.build(DeviceType::CUDA);
    } catch (const std::exception& ex) {
        SKIP(std::string("CUDA graph build failed: ") + ex.what());
    }
    if (!exec) SKIP("CUDA backend not available in this build.");

    std::vector<int32_t> tok_data = {1, 2, 3, 4, 5};
    tokens.copy_from_host(std::as_bytes(std::span(tok_data)));

    std::vector<float> ones_embed(cfg.vocab_size * cfg.hidden_size, 0.1f);
    weights.w_embed.copy_from_host(std::as_bytes(std::span(ones_embed)));

    std::vector<float> ones_lm(cfg.hidden_size * cfg.vocab_size, 0.1f);
    weights.w_lm_head.copy_from_host(std::as_bytes(std::span(ones_lm)));

    std::vector<float> ones_norm(cfg.hidden_size, 1.0f);
    weights.w_final_norm.copy_from_host(std::as_bytes(std::span(ones_norm)));

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        weights.w_attn_norm[l].copy_from_host(std::as_bytes(std::span(ones_norm)));
        weights.w_ffn_norm[l].copy_from_host(std::as_bytes(std::span(ones_norm)));

        std::vector<float> w_mat(cfg.hidden_size * cfg.hidden_size, 0.05f);
        weights.wq[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wk[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wv[l].copy_from_host(std::as_bytes(std::span(w_mat)));
        weights.wo[l].copy_from_host(std::as_bytes(std::span(w_mat)));

        std::vector<float> w_gate_up(cfg.hidden_size * cfg.intermediate_size, 0.05f);
        weights.w_gate[l].copy_from_host(std::as_bytes(std::span(w_gate_up)));
        weights.w_up[l].copy_from_host(std::as_bytes(std::span(w_gate_up)));

        std::vector<float> w_down(cfg.intermediate_size * cfg.hidden_size, 0.05f);
        weights.w_down[l].copy_from_host(std::as_bytes(std::span(w_down)));
    }

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(seq, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf);
    weights.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    weights.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    std::vector<float> mask_buf;
    compute_causal_mask(cfg.num_heads, seq, mask_buf);
    weights.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

    exec->execute();

    std::vector<float> out(seq * cfg.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(out)));

    for (float v : out) {
        CHECK(std::isfinite(v));
    }
}
#endif
