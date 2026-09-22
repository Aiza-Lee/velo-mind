#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "velomind.h"

#include "llama_graph.h"
#include "llama_rope_mask.h"

#include "model.h"
#include "model_loader.h"

namespace {

inline auto bytes_of(const std::vector<float>& v) {
    return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(float));
}

inline auto bytes_of(const std::vector<std::int32_t>& v) {
    return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v.data()), v.size() * sizeof(std::int32_t));
}

inline auto writeable_bytes_of(std::vector<float>& v) {
    return std::span<std::byte>(
        reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(float));
}

}

int main(int argc, char** argv) {
    using namespace velomind;
    using velomind::examples::llama::ForwardWeights;
    using velomind::examples::llama::build_forward_graph_prefill;
    using velomind::examples::llama::compute_rope_cache;
    using velomind::examples::llama::compute_causal_mask;
    using velomind::examples::smollm2::Config;
    using velomind::examples::smollm2::kSmolLM2_135M;
    using velomind::examples::smollm2::bind_smollm2_weights;

    Device device = Device::cpu();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            auto d = Device::from_string(argv[++i]);
            if (d) device = *d;
        }
    }

    if (!device.is_available()) {
        std::cerr << std::format(
            "smollm2_example: target device {} is not available or operational on this system "
            "(check nvidia-smi / GPU drivers, or run with --device ispc / --device cpu)\n",
            device.name());
        return 1;
    }

    const Config cfg = kSmolLM2_135M;

    constexpr std::size_t kSeq = 5;
    std::vector<std::int32_t> token_data = {1, 2, 3, 4, 5};

    Graph g;
    auto tokens = g.input({static_cast<dim_t>(kSeq)}, DataType::Int32);
    ForwardWeights weights = bind_smollm2_weights(g, cfg, kSeq);

    auto logits = build_forward_graph_prefill(g, cfg, tokens, weights);
    if (!logits) {
        std::cerr << "smollm2_example: graph build returned null logits\n";
        return 1;
    }

    std::unique_ptr<Executable> exec;
    try {
        exec = g.build(device);
    } catch (const std::exception& ex) {
        std::cerr << std::format("smollm2_example: graph build on {} failed: {}\n",
                                 device.name(), ex.what());
        return 1;
    }
    if (!exec) {
        std::cerr << "smollm2_example: graph build failed\n";
        return 1;
    }
    std::cout << std::format("compiled smollm2 graph on {}: {} nodes\n",
                             device.name(),
                             exec->num_nodes());

    std::mt19937 rng(0x5011'1A2u);
    auto rand_vec = [&](std::size_t n, float lo, float hi) {
        std::vector<float> v(n);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    tokens.copy_from_host(bytes_of(token_data));

    auto embed = rand_vec(cfg.vocab_size * cfg.hidden_size, -0.05f, 0.05f);
    weights.w_embed.copy_from_host(bytes_of(embed));

    auto lm = rand_vec(cfg.hidden_size * cfg.vocab_size, -0.05f, 0.05f);
    weights.w_lm_head.copy_from_host(bytes_of(lm));

    std::vector<float> norm(cfg.hidden_size, 1.0f);
    weights.w_final_norm.copy_from_host(bytes_of(norm));

    std::vector<float> cos_cache, sin_cache, causal_mask;
    compute_rope_cache(kSeq, cfg.head_dim(), cfg.rope_theta, cos_cache, sin_cache);
    compute_causal_mask(cfg.num_heads, kSeq, causal_mask);
    weights.cos_cache.copy_from_host(bytes_of(cos_cache));
    weights.sin_cache.copy_from_host(bytes_of(sin_cache));
    weights.causal_mask.copy_from_host(bytes_of(causal_mask));

    auto an = rand_vec(cfg.hidden_size,             0.95f, 1.05f);
    auto fn = rand_vec(cfg.hidden_size,             0.95f, 1.05f);
    const std::size_t kv_dim = cfg.effective_num_kv_heads() * cfg.head_dim();
    auto wq = rand_vec(cfg.hidden_size * cfg.hidden_size, -0.05f, 0.05f);
    auto wk = rand_vec(cfg.hidden_size * kv_dim,          -0.05f, 0.05f);
    auto wv = rand_vec(cfg.hidden_size * kv_dim,          -0.05f, 0.05f);
    auto wo = rand_vec(cfg.hidden_size * cfg.hidden_size, -0.05f, 0.05f);
    auto wg = rand_vec(cfg.hidden_size * cfg.intermediate_size, -0.05f, 0.05f);
    auto wu = rand_vec(cfg.hidden_size * cfg.intermediate_size, -0.05f, 0.05f);
    auto wd = rand_vec(cfg.intermediate_size * cfg.hidden_size, -0.05f, 0.05f);

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        weights.w_attn_norm[l].copy_from_host(bytes_of(an));
        weights.w_ffn_norm[l].copy_from_host(bytes_of(fn));
        weights.wq[l].copy_from_host(bytes_of(wq));
        weights.wk[l].copy_from_host(bytes_of(wk));
        weights.wv[l].copy_from_host(bytes_of(wv));
        weights.wo[l].copy_from_host(bytes_of(wo));
        weights.w_gate[l].copy_from_host(bytes_of(wg));
        weights.w_up[l].copy_from_host(bytes_of(wu));
        weights.w_down[l].copy_from_host(bytes_of(wd));
    }

    exec->execute();

    const std::size_t expected_logits = kSeq * cfg.vocab_size;
    std::vector<float> logit_data(expected_logits);
    logits.copy_to_host(writeable_bytes_of(logit_data));

    if (logit_data.size() != expected_logits) {
        std::cerr << std::format(
            "smollm2_example: expected {} logits (seq={}, vocab={}), got {}\n",
            expected_logits, kSeq, cfg.vocab_size, logit_data.size());
        return 1;
    }

    bool any_nan_or_inf = false;
    bool all_zero       = true;
    bool all_equal      = true;
    for (std::size_t i = 0; i < logit_data.size(); ++i) {
        float v = logit_data[i];
        if (!std::isfinite(v)) any_nan_or_inf = true;
        if (v != 0.0f)         all_zero       = false;
        if (v != logit_data[0]) all_equal     = false;
    }

    std::string row;
    const std::size_t last_t = (kSeq - 1) * cfg.vocab_size;
    for (std::size_t i = 0; i < 8 && i < cfg.vocab_size; ++i) {
        if (i > 0) row += ", ";
        row += std::format("{:.4f}", logit_data[last_t + i]);
    }
    std::cout << std::format("logits[seq={}, vocab={}] first 8 of last token: {} ...\n",
                             kSeq, cfg.vocab_size, row);

    if (any_nan_or_inf) {
        std::cerr << "smollm2_example: FAIL — logit contains NaN/Inf\n";
        return 1;
    }
    if (all_zero) {
        std::cerr << "smollm2_example: FAIL — all logits are zero\n";
        return 1;
    }
    if (all_equal) {
        std::cerr << "smollm2_example: FAIL — all logits equal (degenerate forward)\n";
        return 1;
    }

    std::cout << "smollm2_example: PASS\n";
    return 0;
}
