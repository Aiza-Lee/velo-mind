#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <random>
#include <vector>

#include "velomind.h"

#include "llama_graph.h"

#include "model.h"
#include "sample.h"

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
    using velomind::examples::tinyllama::Config;
    using velomind::examples::tinyllama::kTiny;
    using velomind::examples::llama::ForwardWeights;
    using velomind::examples::llama::build_forward_graph_prefill;

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
            "tinyllama_example: target device {} is not available or operational on this system "
            "(check nvidia-smi / GPU drivers, or run with --device ispc / --device cpu)\n",
            device.name());
        return 1;
    }

    constexpr Config C = kTiny;

    constexpr std::size_t kSeq = 3;
    std::vector<std::int32_t> token_data = {0, 1, 2};

    std::mt19937 rng(42);
    auto rand_vec = [&](std::size_t n) {
        std::vector<float> v(n);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    Graph g;
    auto w_embed   = g.input({C.vocab_size, C.hidden_size},    DataType::Float32);
    auto w_lm_head = g.input({C.hidden_size, C.vocab_size},    DataType::Float32);
    auto wq = g.input({C.hidden_size, C.hidden_size},          DataType::Float32);
    auto wk = g.input({C.hidden_size, C.hidden_size},          DataType::Float32);
    auto wv = g.input({C.hidden_size, C.hidden_size},          DataType::Float32);
    auto wo = g.input({C.hidden_size, C.hidden_size},          DataType::Float32);
    auto w_an = g.input({C.hidden_size},                       DataType::Float32);
    auto w_fn = g.input({C.hidden_size},                       DataType::Float32);
    auto w_gate = g.input({C.hidden_size, C.intermediate_size}, DataType::Float32);
    auto w_up   = g.input({C.hidden_size, C.intermediate_size}, DataType::Float32);
    auto w_down = g.input({C.intermediate_size, C.hidden_size}, DataType::Float32);
    auto w_final_norm = g.input({C.hidden_size},               DataType::Float32);

    auto token = g.input({kSeq}, DataType::Int32);

    ForwardWeights weights;
    weights.graph        = &g;
    weights.w_embed      = w_embed;
    weights.w_lm_head    = w_lm_head;
    weights.w_attn_norm  = {w_an};
    weights.wq           = {wq};
    weights.wk           = {wk};
    weights.wv           = {wv};
    weights.wo           = {wo};
    weights.w_ffn_norm   = {w_fn};
    weights.w_gate       = {w_gate};
    weights.w_up         = {w_up};
    weights.w_down       = {w_down};
    weights.w_final_norm = w_final_norm;
    auto logits = build_forward_graph_prefill(g, C, token, weights);

    std::unique_ptr<Executable> exec;
    try {
        exec = g.build(device);
    } catch (const std::exception& ex) {
        std::cerr << std::format("tinyllama_example: graph build on {} failed: {}\n",
                                 device.name(), ex.what());
        return 1;
    }
    if (!exec) {
        std::cerr << "tinyllama_example: graph build failed\n";
        return 1;
    }
    std::cout << std::format("compiled graph on {}: {} nodes\n",
                             device.name(), exec->num_nodes());

    auto fill = [&](Tensor t, std::size_t n) {
        t.copy_from_host(bytes_of(rand_vec(n)));
    };
    fill(w_embed,   C.vocab_size * C.hidden_size);
    fill(w_lm_head, C.hidden_size * C.vocab_size);
    fill(wq, C.hidden_size * C.hidden_size);
    fill(wk, C.hidden_size * C.hidden_size);
    fill(wv, C.hidden_size * C.hidden_size);
    fill(wo, C.hidden_size * C.hidden_size);
    fill(w_an, C.hidden_size);
    fill(w_fn, C.hidden_size);
    fill(w_gate, C.hidden_size * C.intermediate_size);
    fill(w_up,   C.hidden_size * C.intermediate_size);
    fill(w_down, C.intermediate_size * C.hidden_size);
    fill(w_final_norm, C.hidden_size);

    token.copy_from_host(bytes_of(token_data));

    exec->execute();

    std::vector<float> logit_data(logits.numel());
    logits.copy_to_host(writeable_bytes_of(logit_data));
    const std::size_t expected_logits = kSeq * C.vocab_size;
    if (logit_data.size() != expected_logits) {
        std::cerr << std::format(
            "tinyllama_example: expected {} logits (seq={}, vocab={}), got {}\n",
            expected_logits, kSeq, C.vocab_size, logit_data.size());
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

    std::cout << std::format("logits[seq={}, vocab={}] =\n", kSeq, C.vocab_size);
    for (std::size_t t = 0; t < kSeq; ++t) {
        std::string row;
        for (std::size_t i = 0; i < C.vocab_size; ++i) {
            if (i > 0) row += ", ";
            row += std::format("{:.4f}", logit_data[t * C.vocab_size + i]);
        }
        std::cout << std::format("  t={}: {{{}}}\n", t, row);
    }

    if (any_nan_or_inf) {
        std::cerr << "tinyllama_example: FAIL — logit contains NaN/Inf\n";
        return 1;
    }
    if (all_zero) {
        std::cerr << "tinyllama_example: FAIL — all logits are zero\n";
        return 1;
    }
    if (all_equal) {
        std::cerr << "tinyllama_example: FAIL — all logits equal (degenerate forward)\n";
        return 1;
    }

    std::mt19937 sample_rng(12345);
    velomind::examples::tinyllama::SamplerConfig greedy_cfg{.temperature = 0.0f};
    velomind::examples::tinyllama::SamplerConfig topp_cfg{.temperature = 0.8f, .top_p = 0.9f};

    std::int32_t greedy_token = velomind::examples::tinyllama::sample_next_token(
        logits, greedy_cfg, sample_rng);
    std::int32_t topp_token = velomind::examples::tinyllama::sample_next_token(
        logits, topp_cfg, sample_rng);

    std::cout << std::format("sampled next token (argmax): {}\n", greedy_token);
    std::cout << std::format("sampled next token (top-p):  {}\n", topp_token);

    if (greedy_token < 0 || greedy_token >= static_cast<std::int32_t>(C.vocab_size)) {
        std::cerr << "tinyllama_example: FAIL — greedy token out of vocab range\n";
        return 1;
    }
    if (greedy_token != 5) {
        std::cerr << std::format(
            "tinyllama_example: FAIL — expected greedy token 5 (max logit 0.3534 at t=2), got {}\n",
            greedy_token);
        return 1;
    }

    std::cout << "tinyllama_example: PASS\n";
    return 0;
}
