#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/safetensors.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "llama_graph.h"
#include "llama_rope_mask.h"

#include "../examples/smollm2/model.h"
#include "../examples/smollm2/model_loader.h"
#include "../examples/smollm2/smollm2_engine.h"

#include "test_model_assets.h"

using namespace velomind;
using velomind::examples::llama::ForwardWeights;
using velomind::examples::llama::build_forward_graph_prefill;
using velomind::examples::llama::compute_rope_cache;
using velomind::examples::llama::compute_causal_mask;
using velomind::examples::smollm2::Config;
using velomind::examples::smollm2::kSmolLM2_135M;
using velomind::examples::smollm2::bind_smollm2_weights;
using velomind::examples::smollm2::populate_smollm2_weights;
using velomind::examples::smollm2::parse_config_json;

namespace {

    auto load_text_file(const std::filesystem::path& p) -> std::string {
        std::ifstream f(p);
        if (!f) {
            throw std::runtime_error("cannot open " + p.string());
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

}

TEST_CASE("SmolLM2-135M - parse_config_json", "[smollm2]") {
    const auto cfg_path = velomind_test::get_smollm2_model_dir() / "config.json";
    velomind_test::require_or_skip_asset(cfg_path, "SmolLM2 model config");

    const std::string text = load_text_file(cfg_path);
    const Config cfg = parse_config_json(text);

    REQUIRE(cfg.vocab_size        == 49152);
    REQUIRE(cfg.hidden_size       == 576);
    REQUIRE(cfg.intermediate_size == 1536);
    REQUIRE(cfg.num_layers        == 30);
    REQUIRE(cfg.num_heads         == 9);
    REQUIRE(cfg.num_kv_heads      == 3);
    REQUIRE(cfg.head_dim()        == 64);
    REQUIRE(cfg.rms_norm_eps      == Catch::Approx(1e-5f));
    REQUIRE(cfg.rope_theta        == Catch::Approx(100000.0f));
    REQUIRE(cfg.tie_word_embeddings);
}

static auto run_smollm2_synthetic(DeviceType device) -> std::vector<float> {
    const Config cfg = kSmolLM2_135M;
    const std::size_t seq = 5;

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_smollm2_weights(g, cfg, seq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(device);
    REQUIRE(exec);

    std::vector<int32_t> tok_data = {1, 2, 3, 4, 5};
    tokens.copy_from_host(std::as_bytes(std::span(tok_data)));

    std::vector<float> embed(cfg.vocab_size * cfg.hidden_size, 0.05f);
    weights.w_embed.copy_from_host(std::as_bytes(std::span(embed)));

    std::vector<float> lm(cfg.hidden_size * cfg.vocab_size, 0.05f);
    weights.w_lm_head.copy_from_host(std::as_bytes(std::span(lm)));

    std::vector<float> norm(cfg.hidden_size, 1.0f);
    weights.w_final_norm.copy_from_host(std::as_bytes(std::span(norm)));

    std::vector<float> cos_cache, sin_cache, causal_mask;
    compute_rope_cache(seq, cfg.head_dim(), cfg.rope_theta, cos_cache, sin_cache);
    compute_causal_mask(cfg.num_heads, seq, causal_mask);
    weights.cos_cache.copy_from_host(std::as_bytes(std::span(cos_cache)));
    weights.sin_cache.copy_from_host(std::as_bytes(std::span(sin_cache)));
    weights.causal_mask.copy_from_host(std::as_bytes(std::span(causal_mask)));

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        std::vector<float> an(cfg.hidden_size, 1.0f);
        weights.w_attn_norm[l].copy_from_host(std::as_bytes(std::span(an)));
        std::vector<float> fn(cfg.hidden_size, 1.0f);
        weights.w_ffn_norm[l].copy_from_host(std::as_bytes(std::span(fn)));
        const std::size_t kv_dim = cfg.effective_num_kv_heads() * cfg.head_dim();
        std::vector<float> wq(cfg.hidden_size * cfg.hidden_size, 0.05f);
        weights.wq[l].copy_from_host(std::as_bytes(std::span(wq)));
        std::vector<float> wk(cfg.hidden_size * kv_dim, 0.05f);
        weights.wk[l].copy_from_host(std::as_bytes(std::span(wk)));
        std::vector<float> wv(cfg.hidden_size * kv_dim, 0.05f);
        weights.wv[l].copy_from_host(std::as_bytes(std::span(wv)));
        std::vector<float> wo(cfg.hidden_size * cfg.hidden_size, 0.05f);
        weights.wo[l].copy_from_host(std::as_bytes(std::span(wo)));
        std::vector<float> wg(cfg.hidden_size * cfg.intermediate_size, 0.05f);
        weights.w_gate[l].copy_from_host(std::as_bytes(std::span(wg)));
        std::vector<float> wu(cfg.hidden_size * cfg.intermediate_size, 0.05f);
        weights.w_up[l].copy_from_host(std::as_bytes(std::span(wu)));
        std::vector<float> wd(cfg.intermediate_size * cfg.hidden_size, 0.05f);
        weights.w_down[l].copy_from_host(std::as_bytes(std::span(wd)));
    }

    exec->execute();


    std::vector<float> out(seq * cfg.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(out)));

    for (float v : out) {
        CHECK(std::isfinite(v));
    }
    return out;
}

TEST_CASE("SmolLM2-135M - 5-token prefill synthetic CPU", "[smollm2][cpu]") {
    run_smollm2_synthetic(DeviceType::CPU);
}
#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("SmolLM2-135M - 5-token prefill synthetic ISPC", "[smollm2][ispc]") {
    run_smollm2_synthetic(DeviceType::ISPC);
}
#endif
#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("SmolLM2-135M - 5-token prefill synthetic CUDA", "[smollm2][cuda]") {
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");
    run_smollm2_synthetic(DeviceType::CUDA);
}
#endif
#ifdef VELOMIND_ENABLE_VULKAN
TEST_CASE("SmolLM2-135M - 5-token prefill synthetic Vulkan", "[smollm2][vulkan]") {
    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");
    run_smollm2_synthetic(DeviceType::VULKAN);
}
#endif

static void check_smollm2_reference(DeviceType device) {
    const auto ref_path = velomind_test::require_or_skip_asset(
        "examples/smollm2/testdata/reference_prompt_5tokens.safetensors", "SmolLM2 HF reference data");

    auto ref_file = load_safetensors(ref_path);
    REQUIRE(ref_file.has_value());

    const auto& ref_tensors = ref_file->tensors;
    REQUIRE(ref_tensors.count("input_tokens"));
    REQUIRE(ref_tensors.count("reference_logits"));

    const auto model_dir = velomind_test::get_smollm2_model_dir();
    const auto cfg_path = model_dir / "config.json";
    velomind_test::require_or_skip_asset(cfg_path, "SmolLM2 model config");

    const auto model_path = model_dir / "model.safetensors";
    velomind_test::require_or_skip_asset(model_path, "SmolLM2 model weights");

    const std::string cfg_text = load_text_file(cfg_path);
    const Config cfg = parse_config_json(cfg_text);

    auto model_sf = load_safetensors(model_path);
    REQUIRE(model_sf.has_value());

    const std::size_t seq = 5;
    Graph g;
    auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto weights = bind_smollm2_weights(g, cfg, seq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    REQUIRE(logits);

    auto exec = g.build(device);
    REQUIRE(exec);

    populate_smollm2_weights(*model_sf, cfg, weights, seq);

    const auto& tok_storage = ref_tensors.at("input_tokens");
    tokens.copy_from_host(std::span<const std::byte>(
        static_cast<const std::byte*>(tok_storage->data), tok_storage->size_bytes));

    exec->execute();

    std::vector<float> actual_logits(seq * cfg.vocab_size);
    logits.copy_to_host(std::as_writable_bytes(std::span(actual_logits)));

    const auto& ref_logits_storage = ref_tensors.at("reference_logits");
    const float* ref_data = static_cast<const float*>(ref_logits_storage->data);

    velomind_test::require_all_finite(actual_logits, "SmolLM2 actual_logits");
    velomind_test::require_all_finite(ref_data, actual_logits.size(), "SmolLM2 reference_logits");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < actual_logits.size(); ++i) {
        float diff = std::abs(actual_logits[i] - ref_data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    CHECK(max_diff < 1e-2f);

    std::vector<float> last_token_logits(cfg.vocab_size);
    for (std::size_t v = 0; v < cfg.vocab_size; ++v) {
        last_token_logits[v] = actual_logits[(seq - 1) * cfg.vocab_size + v];
    }
    int32_t predicted_token = 0;
    float best = last_token_logits[0];
    for (std::size_t v = 1; v < cfg.vocab_size; ++v) {
        if (last_token_logits[v] > best) {
            best = last_token_logits[v];
            predicted_token = static_cast<int32_t>(v);
        }
    }
    CHECK(std::isfinite(best));
    CHECK(predicted_token >= 0);
    CHECK(predicted_token < static_cast<int32_t>(cfg.vocab_size));
}

TEST_CASE("SmolLM2-135M - 5-token prefill CPU vs HuggingFace reference",
          "[.][smollm2][slow][cpu]") {
    check_smollm2_reference(DeviceType::CPU);
}
#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("SmolLM2-135M - 5-token prefill ISPC vs HuggingFace reference",
          "[.][smollm2][slow][ispc]") {
    check_smollm2_reference(DeviceType::ISPC);
}
#endif
#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("SmolLM2-135M - 5-token prefill CUDA vs HuggingFace reference",
          "[.][smollm2][slow][cuda]") {
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");
    check_smollm2_reference(DeviceType::CUDA);
}
#endif
#ifdef VELOMIND_ENABLE_VULKAN
TEST_CASE("SmolLM2-135M - 5-token prefill Vulkan vs HuggingFace reference",
          "[.][smollm2][slow][vulkan]") {
    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");
    check_smollm2_reference(DeviceType::VULKAN);
}
#endif

TEST_CASE("SmolLM2-135M - Native GQA weight dimensions and memory savings", "[smollm2][gqa]") {
    const Config cfg = kSmolLM2_135M;
    Graph g;
    auto weights = bind_smollm2_weights(g, cfg, 0);

    REQUIRE(cfg.num_heads == 9);
    REQUIRE(cfg.num_kv_heads == 3);
    REQUIRE(cfg.head_dim() == 64);
    REQUIRE(cfg.hidden_size == 576);

    const dim_t expected_q_dim  = static_cast<dim_t>(cfg.hidden_size);
    const dim_t expected_kv_dim = static_cast<dim_t>(cfg.num_kv_heads * cfg.head_dim());

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        REQUIRE(weights.wq[l].shape() == shape_t{expected_q_dim, expected_q_dim});
        REQUIRE(weights.wk[l].shape() == shape_t{expected_q_dim, expected_kv_dim});
        REQUIRE(weights.wv[l].shape() == shape_t{expected_q_dim, expected_kv_dim});
        REQUIRE(weights.wo[l].shape() == shape_t{expected_q_dim, expected_q_dim});
    }

    // 内存削减量验证：原始展开权重 576x576，原生保留 576x192，削减 66.67%
    const std::size_t raw_kv_bytes_per_layer = 2 * cfg.hidden_size * expected_kv_dim * sizeof(float);
    const std::size_t expanded_kv_bytes_per_layer = 2 * cfg.hidden_size * cfg.hidden_size * sizeof(float);
    const std::size_t total_saved_bytes = cfg.num_layers * (expanded_kv_bytes_per_layer - raw_kv_bytes_per_layer);

    REQUIRE(raw_kv_bytes_per_layer == 884736);
    REQUIRE(expanded_kv_bytes_per_layer == 2654208);
    REQUIRE(total_saved_bytes == 53084160);
}

TEST_CASE("SmolLM2-135M - Native GQA KV cache shape and incremental decode alignment", "[smollm2][gqa][decode]") {
    using namespace velomind::examples::smollm2;

    EngineConfig ecfg;
    ecfg.use_synthetic = true;
    ecfg.device = DeviceType::CPU;
    ecfg.model_config = kSmolLM2_135M;
    ecfg.model_config.num_layers = 2; // 使用 2 层合成小模型进行高速端到端验证

    SmolLM2Engine engine(ecfg);
    engine.load();
    REQUIRE(engine.is_loaded());

    std::vector<std::int32_t> prompt = {101, 102, 103, 104, 105};
    std::vector<std::int32_t> accumulated = prompt;

    engine.reset_session();
    auto prefill_logits = engine.prefill(prompt);
    auto ref_prefill_logits = engine.forward_logits(prompt);

    REQUIRE(prefill_logits.size() == ref_prefill_logits.size());
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < prefill_logits.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(prefill_logits[i] - ref_prefill_logits[i]));
    }
    REQUIRE(max_diff < 1e-4f);

    // 验证 KV cache 中首维严格为 num_kv_heads (3)，而非展开的 num_heads (9)
    const auto& session_kv = engine.inner().kv_cache();
    REQUIRE(session_kv.seq_len == 5);
    REQUIRE(session_kv.layers.size() == 2);
    for (std::size_t l = 0; l < 2; ++l) {
        REQUIRE(session_kv.layers[l].k->shape == shape_t{3, 5, 64});
        REQUIRE(session_kv.layers[l].v->shape == shape_t{3, 5, 64});
    }

    auto argmax_fn = [](std::span<const float> l) -> std::int32_t {
        return static_cast<std::int32_t>(
            std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    };

    std::int32_t next_token = argmax_fn(prefill_logits);
    std::int32_t ref_next   = argmax_fn(ref_prefill_logits);
    REQUIRE(next_token == ref_next);

    // 增量自回归解码 4 步
    const std::size_t decode_steps = 4;
    for (std::size_t step = 0; step < decode_steps; ++step) {
        accumulated.push_back(next_token);
        auto decode_logits = engine.forward_step(next_token, engine.session_seq_len());
        auto ref_step_logits = engine.forward_logits(accumulated);

        REQUIRE(decode_logits.size() == ref_step_logits.size());
        float step_diff = 0.0f;
        for (std::size_t i = 0; i < decode_logits.size(); ++i) {
            step_diff = std::max(step_diff, std::abs(decode_logits[i] - ref_step_logits[i]));
        }
        REQUIRE(step_diff < 1e-4f);

        std::int32_t cand     = argmax_fn(decode_logits);
        std::int32_t cand_ref = argmax_fn(ref_step_logits);
        REQUIRE(cand == cand_ref);

        // 验证步进后 KV cache 的序列长度扩展与头数保持原生 3
        const auto& kv_step = engine.inner().kv_cache();
        const dim_t expected_seq = static_cast<dim_t>(prompt.size() + step + 1);
        REQUIRE(kv_step.seq_len == static_cast<std::size_t>(expected_seq));
        for (std::size_t l = 0; l < 2; ++l) {
            REQUIRE(kv_step.layers[l].k->shape == shape_t{3, expected_seq, 64});
            REQUIRE(kv_step.layers[l].v->shape == shape_t{3, expected_seq, 64});
        }

        next_token = cand;
    }
}
