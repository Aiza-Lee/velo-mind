#include "llama_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/safetensors.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "llama_graph.h"
#include "llama_rope_mask.h"

namespace velomind::examples::llama {

using velomind::DataType;
using velomind::DeviceType;
using velomind::dim_t;
using velomind::Executable;
using velomind::Graph;
using velomind::SafetensorsFile;
using velomind::Tensor;
using velomind::TensorStorage;

LlamaTextStreamer::LlamaTextStreamer(
    const TokenizerAdapter&              tokenizer,
    std::function<void(const std::string&)> callback)
    : tokenizer_(tokenizer), callback_(std::move(callback)) {}

void LlamaTextStreamer::reset(std::span<const std::int32_t> initial_tokens) {
    all_tokens_.assign(initial_tokens.begin(), initial_tokens.end());
    prompt_tokens_count_ = initial_tokens.size();
    generated_tokens_.clear();
    full_generated_text_.clear();
    if (tokenizer_.loaded() && !all_tokens_.empty()) {
        decoded_all_ = tokenizer_.decode(all_tokens_);
    } else {
        decoded_all_.clear();
    }
}

auto LlamaTextStreamer::put(std::int32_t token_id) -> std::string {
    all_tokens_.push_back(token_id);
    generated_tokens_.push_back(token_id);

    if (!tokenizer_.loaded()) {
        std::string fallback = "[" + std::to_string(token_id) + "]";
        full_generated_text_ += fallback;
        if (callback_) callback_(fallback);
        return fallback;
    }

    std::string new_decoded = tokenizer_.decode(all_tokens_);
    std::string delta;

    if (new_decoded.size() >= decoded_all_.size() &&
        new_decoded.compare(0, decoded_all_.size(), decoded_all_) == 0) {
        delta = new_decoded.substr(decoded_all_.size());
    } else {
        std::size_t prefix_len = 0;
        while (prefix_len < decoded_all_.size() &&
               prefix_len < new_decoded.size() &&
               decoded_all_[prefix_len] == new_decoded[prefix_len]) {
            ++prefix_len;
        }
        if (new_decoded.size() > prefix_len) {
            delta = new_decoded.substr(prefix_len);
        }
    }

    decoded_all_ = std::move(new_decoded);
    full_generated_text_ += delta;
    if (callback_ && !delta.empty()) {
        callback_(delta);
    }
    return delta;
}

auto LlamaTextStreamer::flush() -> std::string {
    if (!tokenizer_.loaded() || all_tokens_.empty()) {
        return "";
    }
    std::string final_decoded = tokenizer_.decode(all_tokens_);
    std::string delta;
    if (final_decoded.size() > decoded_all_.size()) {
        delta = final_decoded.substr(decoded_all_.size());
        full_generated_text_ += delta;
        decoded_all_ = std::move(final_decoded);
        if (callback_ && !delta.empty()) {
            callback_(delta);
        }
    }
    return delta;
}

LlamaEngine::LlamaEngine(LlamaEngineConfig  config,
                         BindWeightsFn      bind_weights,
                         PopulateWeightsFn  populate_weights,
                         FillSyntheticFn    fill_synthetic,
                         TokenizerFactory   tokenizer_factory)
    : config_(std::move(config)),
      bind_weights_(std::move(bind_weights)),
      populate_weights_(std::move(populate_weights)),
      fill_synthetic_(std::move(fill_synthetic)),
      tokenizer_factory_(std::move(tokenizer_factory)) {}

LlamaEngine::~LlamaEngine() = default;

void LlamaEngine::record_weight_storages_(
    const ForwardWeights& w,
    std::size_t num_layers
) {
    w_embed_storage_      = w.w_embed.shared_storage();
    w_lm_head_storage_    = w.w_lm_head.shared_storage();
    w_final_norm_storage_ = w.w_final_norm.shared_storage();

    w_attn_norm_storage_.clear();
    w_ffn_norm_storage_.clear();
    wq_storage_.clear();
    wk_storage_.clear();
    wv_storage_.clear();
    wo_storage_.clear();
    w_gate_storage_.clear();
    w_up_storage_.clear();
    w_down_storage_.clear();

    w_attn_norm_storage_.reserve(num_layers);
    w_ffn_norm_storage_.reserve(num_layers);
    wq_storage_.reserve(num_layers);
    wk_storage_.reserve(num_layers);
    wv_storage_.reserve(num_layers);
    wo_storage_.reserve(num_layers);
    w_gate_storage_.reserve(num_layers);
    w_up_storage_.reserve(num_layers);
    w_down_storage_.reserve(num_layers);

    for (std::size_t l = 0; l < num_layers; ++l) {
        w_attn_norm_storage_.push_back(w.w_attn_norm[l].shared_storage());
        w_ffn_norm_storage_.push_back(w.w_ffn_norm[l].shared_storage());
        wq_storage_.push_back(w.wq[l].shared_storage());
        wk_storage_.push_back(w.wk[l].shared_storage());
        wv_storage_.push_back(w.wv[l].shared_storage());
        wo_storage_.push_back(w.wo[l].shared_storage());
        w_gate_storage_.push_back(w.w_gate[l].shared_storage());
        w_up_storage_.push_back(w.w_up[l].shared_storage());
        w_down_storage_.push_back(w.w_down[l].shared_storage());
    }
}

void LlamaEngine::load_synthetic_weights_() {
    namespace fs = std::filesystem;

    if (!tokenizer_->loaded()) {
        const std::vector<fs::path> candidates = {
            fs::path{config_.tokenizer_path},
            fs::path{"examples/tinyllama/testdata/test.model"},
            fs::path{"../examples/tinyllama/testdata/test.model"},
            fs::path{"/home/aiza/workspace/dev/velo-mind/examples/tinyllama/testdata/test.model"}
        };
        for (const auto& c : candidates) {
            if (fs::exists(c)) {
                tokenizer_->load(c.string());
                break;
            }
        }
    }

    const auto& cfg = config_.model_config;
    Graph init_g;
    auto w = bind_weights_(init_g, cfg, 0);
    auto exec = init_g.build(config_.device);
    if (!exec) {
        throw std::runtime_error("LlamaEngine: failed to build synthetic weight graph");
    }

    if (fill_synthetic_) {
        fill_synthetic_(init_g, w, cfg);
    } else {
        std::mt19937 rng(42);
        auto rand_vec = [&](std::size_t n, float scale) {
            std::vector<float> v(n);
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (auto& x : v) x = dist(rng);
            return v;
        };
        auto fill = [&](Tensor t, std::size_t n, float scale) {
            auto v = rand_vec(n, scale);
            t.copy_from_host(std::as_bytes(std::span(v)));
        };

        fill(w.w_embed,      cfg.vocab_size * cfg.hidden_size, 0.1f);
        fill(w.w_lm_head,    cfg.hidden_size * cfg.vocab_size, 0.1f);
        fill(w.w_final_norm, cfg.hidden_size,                  1.0f);
        const std::size_t kv_dim = cfg.effective_num_kv_heads() * cfg.head_dim();
        for (std::size_t l = 0; l < cfg.num_layers; ++l) {
            fill(w.w_attn_norm[l], cfg.hidden_size,                  1.0f);
            fill(w.w_ffn_norm[l],  cfg.hidden_size,                  1.0f);
            fill(w.wq[l], cfg.hidden_size * cfg.hidden_size,         0.05f);
            fill(w.wk[l], cfg.hidden_size * kv_dim,                  0.05f);
            fill(w.wv[l], cfg.hidden_size * kv_dim,                  0.05f);
            fill(w.wo[l], cfg.hidden_size * cfg.hidden_size,         0.05f);
            fill(w.w_gate[l], cfg.hidden_size * cfg.intermediate_size, 0.05f);
            fill(w.w_up[l],   cfg.hidden_size * cfg.intermediate_size, 0.05f);
            fill(w.w_down[l], cfg.intermediate_size * cfg.hidden_size, 0.05f);
        }
    }

    record_weight_storages_(w, cfg.num_layers);
}

void LlamaEngine::load_safetensors_weights_() {
    namespace fs = std::filesystem;

    if (!fs::exists(config_.model_path)) {
        throw std::runtime_error("LlamaEngine: model file does not exist: " + config_.model_path);
    }
    if (!fs::exists(config_.tokenizer_path)) {
        throw std::runtime_error("LlamaEngine: tokenizer file does not exist: " + config_.tokenizer_path);
    }

    tokenizer_->load(config_.tokenizer_path);

    auto sf = load_safetensors(config_.model_path);
    if (!sf.has_value()) {
        throw std::runtime_error("LlamaEngine: failed to load safetensors from: " + config_.model_path);
    }

    const auto& cfg = config_.model_config;
    Graph init_g;
    auto w = bind_weights_(init_g, cfg, 0);
    auto exec = init_g.build(config_.device);
    if (!exec) {
        throw std::runtime_error("LlamaEngine: failed to build initial weight graph");
    }
    populate_weights_(*sf, cfg, w, 0);

    record_weight_storages_(w, cfg.num_layers);
}

void LlamaEngine::load() {
    Device dev(config_.device);
    if (!dev.is_available()) {
        throw std::runtime_error(
            std::string("LlamaEngine: target device [") +
            dev.name() +
            "] is not available or operational on this system "
            "(check drivers / nvidia-smi)");
    }

    if (tokenizer_factory_ == nullptr) {
        throw std::runtime_error("LlamaEngine: tokenizer_factory is null");
    }
    tokenizer_ = tokenizer_factory_();
    if (tokenizer_ == nullptr) {
        throw std::runtime_error("LlamaEngine: tokenizer_factory returned null");
    }

    if (config_.use_synthetic) {
        load_synthetic_weights_();
    } else {
        load_safetensors_weights_();
    }
    loaded_ = true;
}

void LlamaEngine::bind_model_weights(Graph& g, ForwardWeights& step_w) const {
    const auto& cfg = config_.model_config;
    step_w.graph = &g;
    step_w.w_embed      = g.input(w_embed_storage_);
    step_w.w_lm_head    = g.input(w_lm_head_storage_);
    step_w.w_final_norm = g.input(w_final_norm_storage_);

    step_w.w_attn_norm.reserve(cfg.num_layers);
    step_w.w_ffn_norm.reserve(cfg.num_layers);
    step_w.wq.reserve(cfg.num_layers);
    step_w.wk.reserve(cfg.num_layers);
    step_w.wv.reserve(cfg.num_layers);
    step_w.wo.reserve(cfg.num_layers);
    step_w.w_gate.reserve(cfg.num_layers);
    step_w.w_up.reserve(cfg.num_layers);
    step_w.w_down.reserve(cfg.num_layers);

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        step_w.w_attn_norm.push_back(g.input(w_attn_norm_storage_[l]));
        step_w.w_ffn_norm.push_back(g.input(w_ffn_norm_storage_[l]));
        step_w.wq.push_back(g.input(wq_storage_[l]));
        step_w.wk.push_back(g.input(wk_storage_[l]));
        step_w.wv.push_back(g.input(wv_storage_[l]));
        step_w.wo.push_back(g.input(wo_storage_[l]));
        step_w.w_gate.push_back(g.input(w_gate_storage_[l]));
        step_w.w_up.push_back(g.input(w_up_storage_[l]));
        step_w.w_down.push_back(g.input(w_down_storage_[l]));
    }
}

void LlamaEngine::reset_session() noexcept {
    session_kv_cache_.clear();
}

auto LlamaEngine::session_seq_len() const noexcept -> std::size_t {
    return session_kv_cache_.seq_len;
}

auto LlamaEngine::kv_cache() const noexcept -> const SessionKVCache& {
    return session_kv_cache_;
}

auto LlamaEngine::prefill(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    if (!loaded_) {
        throw std::runtime_error("LlamaEngine::prefill: engine not loaded");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("LlamaEngine::prefill: tokens is empty");
    }

    const std::size_t S = tokens.size();
    const auto& cfg = config_.model_config;

    Graph g;
    auto tokens_t = g.input({static_cast<dim_t>(S)}, DataType::Int32);

    ForwardWeights step_w;
    bind_model_weights(g, step_w);

    const dim_t S_dim  = static_cast<dim_t>(S);
    const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
    const dim_t NH_dim = static_cast<dim_t>(cfg.num_heads);

    step_w.cos_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.sin_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.causal_mask = g.input({NH_dim, S_dim, S_dim}, DataType::Float32);

    KVCacheHandles kv_out;
    auto logits_t = build_forward_graph_prefill(g, cfg, tokens_t, step_w, &kv_out, true);
    auto exec = g.build(config_.device);
    if (!exec) {
        throw std::runtime_error("LlamaEngine::prefill: graph compilation failed");
    }

    tokens_t.copy_from_host(std::as_bytes(tokens));

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(S, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf, 0);
    step_w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    step_w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    std::vector<float> mask_buf;
    compute_causal_mask(cfg.num_heads, S, mask_buf);
    step_w.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

    exec->execute();

    session_kv_cache_.layers.resize(cfg.num_layers);
    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        session_kv_cache_.layers[l].k = kv_out.k[l].shared_storage();
        session_kv_cache_.layers[l].v = kv_out.v[l].shared_storage();
    }
    session_kv_cache_.seq_len = S;

    const std::size_t vocab = static_cast<std::size_t>(logits_t.shape()[1]);
    std::vector<float> last(vocab);
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(last.data()),
        last.size() * sizeof(float));
    logits_t.copy_to_host(byte_span);
    return last;
}

auto LlamaEngine::forward_step(std::int32_t token, std::size_t pos)
    -> std::vector<float> {
    if (!loaded_) {
        throw std::runtime_error("LlamaEngine::forward_step: engine not loaded");
    }
    if (session_kv_cache_.empty()) {
        std::array<std::int32_t, 1> single_tok = {token};
        return prefill(single_tok);
    }

    const auto& cfg = config_.model_config;

    Graph g;
    auto token_t = g.input({1}, DataType::Int32);

    ForwardWeights step_w;
    bind_model_weights(g, step_w);

    const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
    step_w.cos_cache   = g.input({1, HD_dim}, DataType::Float32);
    step_w.sin_cache   = g.input({1, HD_dim}, DataType::Float32);

    KVCacheHandles kv_in;
    kv_in.k.reserve(cfg.num_layers);
    kv_in.v.reserve(cfg.num_layers);
    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        kv_in.k.push_back(g.input(session_kv_cache_.layers[l].k));
        kv_in.v.push_back(g.input(session_kv_cache_.layers[l].v));
    }

    KVCacheHandles kv_out;
    auto logits_t = build_forward_graph_decode(g, cfg, token_t, step_w, kv_in, &kv_out);
    auto exec = g.build(config_.device);
    if (!exec) {
        throw std::runtime_error("LlamaEngine::forward_step: graph compilation failed");
    }

    std::array<std::int32_t, 1> tok_buf = {token};
    token_t.copy_from_host(std::as_bytes(std::span(tok_buf)));

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(1, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf, pos);
    step_w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    step_w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    exec->execute();

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        session_kv_cache_.layers[l].k = kv_out.k[l].shared_storage();
        session_kv_cache_.layers[l].v = kv_out.v[l].shared_storage();
    }
    session_kv_cache_.seq_len = pos + 1;

    const std::size_t vocab = static_cast<std::size_t>(logits_t.shape()[1]);
    std::vector<float> last(vocab);
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(last.data()),
        last.size() * sizeof(float));
    logits_t.copy_to_host(byte_span);
    return last;
}

auto LlamaEngine::forward_logits(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    if (!loaded_) {
        throw std::runtime_error("LlamaEngine::forward_logits: engine not loaded");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("LlamaEngine::forward_logits: tokens is empty");
    }

    const std::size_t S = tokens.size();
    const auto& cfg = config_.model_config;

    // 权重存储跨步共享；激活、RoPE 和因果掩码按当前序列长度重建。
    Graph g;
    auto tokens_t = g.input({static_cast<dim_t>(S)}, DataType::Int32);

    ForwardWeights step_w;
    bind_model_weights(g, step_w);

    const dim_t S_dim  = static_cast<dim_t>(S);
    const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
    const dim_t NH_dim = static_cast<dim_t>(cfg.num_heads);

    step_w.cos_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.sin_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.causal_mask = g.input({NH_dim, S_dim, S_dim}, DataType::Float32);

    auto logits_t = build_forward_graph_prefill(g, cfg, tokens_t, step_w);
    auto exec = g.build(config_.device);
    if (!exec) {
        throw std::runtime_error(
            "LlamaEngine::forward_logits: graph compilation failed");
    }

    tokens_t.copy_from_host(std::as_bytes(tokens));

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(S, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf, 0);
    step_w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    step_w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    std::vector<float> mask_buf;
    compute_causal_mask(cfg.num_heads, S, mask_buf);
    step_w.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

    exec->execute();

    if (logits_t.shape().size() != 2) {
        throw std::runtime_error(
            "LlamaEngine::forward_logits: expected 2-D logits tensor");
    }
    const std::size_t seq_dim = static_cast<std::size_t>(logits_t.shape()[0]);
    const std::size_t vocab   = static_cast<std::size_t>(logits_t.shape()[1]);
    if (seq_dim != S) {
        throw std::runtime_error(
            "LlamaEngine::forward_logits: logits seq dim != input S");
    }
    // 构图输出全序列 logits，但读取时使用局部 D2H 仅搬运末行数据，避免分配与复制 S*vocab 冗余内存。
    std::vector<float> last(vocab);
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(last.data()),
        last.size() * sizeof(float));
    logits_t.copy_to_host(byte_span, (seq_dim - 1) * vocab * sizeof(float));
    return last;
}

auto LlamaEngine::forward_full_logits(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    if (!loaded_) {
        throw std::runtime_error("LlamaEngine::forward_full_logits: engine not loaded");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("LlamaEngine::forward_full_logits: tokens is empty");
    }

    const std::size_t S = tokens.size();
    const auto& cfg = config_.model_config;

    Graph g;
    auto tokens_t = g.input({static_cast<dim_t>(S)}, DataType::Int32);

    ForwardWeights step_w;
    bind_model_weights(g, step_w);

    const dim_t S_dim  = static_cast<dim_t>(S);
    const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
    const dim_t NH_dim = static_cast<dim_t>(cfg.num_heads);

    step_w.cos_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.sin_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
    step_w.causal_mask = g.input({NH_dim, S_dim, S_dim}, DataType::Float32);

    auto logits_t = build_forward_graph_prefill(g, cfg, tokens_t, step_w, nullptr, false);
    auto exec = g.build(config_.device);
    if (!exec) {
        throw std::runtime_error(
            "LlamaEngine::forward_full_logits: graph compilation failed");
    }

    tokens_t.copy_from_host(std::as_bytes(tokens));

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(S, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf, 0);
    step_w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
    step_w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));

    std::vector<float> mask_buf;
    compute_causal_mask(cfg.num_heads, S, mask_buf);
    step_w.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

    exec->execute();

    if (logits_t.shape().size() != 2) {
        throw std::runtime_error(
            "LlamaEngine::forward_full_logits: expected 2-D logits tensor");
    }
    const std::size_t seq_dim = static_cast<std::size_t>(logits_t.shape()[0]);
    const std::size_t vocab   = static_cast<std::size_t>(logits_t.shape()[1]);
    if (seq_dim != S) {
        throw std::runtime_error(
            "LlamaEngine::forward_full_logits: logits seq dim != input S");
    }
    std::vector<float> full_buf(seq_dim * vocab);
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(full_buf.data()),
        full_buf.size() * sizeof(float));
    logits_t.copy_to_host(byte_span);
    return full_buf;
}

auto LlamaEngine::generate(
    const std::string&                            prompt,
    const std::function<std::int32_t(
        std::span<const float>)>&                 sample_fn,
    std::size_t                                   max_new_tokens,
    std::uint32_t                                 seed,
    std::function<void(std::int32_t, const std::string&)> on_token)
    -> LlamaGenerationResult {
    std::vector<std::int32_t> prompt_tokens;
    if (tokenizer_->loaded()) {
        auto enc = tokenizer_->encode(prompt);
        if (tokenizer_->bos_id() >= 0 &&
            (enc.empty() || enc.front() != tokenizer_->bos_id())) {
            prompt_tokens.push_back(tokenizer_->bos_id());
        }
        prompt_tokens.insert(prompt_tokens.end(), enc.begin(), enc.end());
    } else {
        prompt_tokens = {1};
    }
    return generate(std::span<const std::int32_t>(prompt_tokens),
                    sample_fn, max_new_tokens, seed, std::move(on_token));
}

auto LlamaEngine::generate(
    std::span<const std::int32_t>                 prompt_tokens,
    const std::function<std::int32_t(
        std::span<const float>)>&                 sample_fn,
    std::size_t                                   max_new_tokens,
    std::uint32_t                                 seed,
    std::function<void(std::int32_t, const std::string&)> on_token)
    -> LlamaGenerationResult {
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("LlamaEngine::generate: prompt_tokens is empty");
    }
    LlamaGenerationResult result;
    result.prompt_token_count = prompt_tokens.size();

    LlamaTextStreamer streamer(*tokenizer_);
    streamer.reset(prompt_tokens);

    std::mt19937 rng(seed);
    const std::int32_t eos_id = tokenizer_->loaded() ? tokenizer_->eos_id() : -1;

    auto t_start = std::chrono::steady_clock::now();

    // 增量 prefill：执行首次上下文预填充并建立 KV cache。
    reset_session();
    auto last_logits = prefill(prompt_tokens);
    if (last_logits.empty() || max_new_tokens == 0) {
        return result;
    }

    std::int32_t next_token = sample_fn(last_logits);
    result.generated_tokens.push_back(next_token);
    std::string piece = streamer.put(next_token);
    if (on_token) on_token(next_token, piece);

    // 增量 decode：仅前向单 token，结合逐层 KV cache 生成后续 token。
    for (std::size_t step = 1; step < max_new_tokens; ++step) {
        if (eos_id >= 0 && next_token == eos_id) break;
        auto logits = forward_step(next_token, session_kv_cache_.seq_len);
        if (logits.empty()) break;
        next_token = sample_fn(logits);
        result.generated_tokens.push_back(next_token);
        piece = streamer.put(next_token);
        if (on_token) on_token(next_token, piece);
    }
    std::string remaining = streamer.flush();
    if (on_token && !remaining.empty()) on_token(-1, remaining);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    result.text = streamer.text();
    result.elapsed_seconds = elapsed;
    result.tokens_per_second =
        (elapsed > 0.0 && !result.generated_tokens.empty())
            ? static_cast<double>(result.generated_tokens.size()) / elapsed
            : 0.0;
    return result;
}

namespace {

struct ConsoleState {
    float     temperature      = 0.7f;
    float     top_p            = 0.9f;
    std::size_t max_new_tokens = 64;
    std::uint32_t seed         = 42;

    std::string system_prompt;

    std::vector<std::string> turns;
    bool                     have_system = false;

    std::size_t last_tokens = 0;
    double      last_seconds = 0.0;
    double      last_tps     = 0.0;
    std::string last_text;
};

auto make_default_sampler(const ConsoleState& state)
    -> std::function<std::int32_t(std::span<const float>)> {
    return [state](std::span<const float> logits) -> std::int32_t {
        if (logits.empty()) return -1;
        if (state.temperature <= 0.0f || state.top_p <= 0.0f) {

            std::int32_t best = -1;
            float bv = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < logits.size(); ++i) {
                float v = logits[i];
                if (std::isnan(v)) continue;
                if (best == -1 || v > bv) { bv = v; best = static_cast<std::int32_t>(i); }
            }
            return best;
        }

        float max_v = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (float v : logits) {
            if (std::isfinite(v) && (!any || v > max_v)) { max_v = v; any = true; }
        }
        if (!any) return -1;
        const float inv_t = 1.0f / state.temperature;
        struct Prob { float p; std::int32_t id; };
        std::vector<Prob> cands;
        cands.reserve(logits.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            float v = logits[i];
            if (std::isnan(v)) continue;
            float p = std::exp((v - max_v) * inv_t);
            sum += p;
            cands.push_back({p, static_cast<std::int32_t>(i)});
        }
        if (cands.empty()) return -1;
        const float inv_sum = static_cast<float>(1.0 / sum);
        for (auto& c : cands) c.p *= inv_sum;
        std::sort(cands.begin(), cands.end(),
                  [](const Prob& a, const Prob& b) { return a.p > b.p; });
        float cum = 0.0f;
        std::size_t cutoff = 0;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].p;
            cutoff = i + 1;
            if (cum >= state.top_p) break;
        }
        if (cutoff == 0 && !cands.empty()) {
            cutoff = 1;
            cum = cands[0].p;
        }
        std::mt19937 rng(static_cast<std::uint32_t>(std::random_device{}()));
        std::uniform_real_distribution<float> dist(0.0f, cum);
        float r = dist(rng);
        float acc = 0.0f;
        for (std::size_t i = 0; i < cutoff; ++i) {
            acc += cands[i].p;
            if (r <= acc) return cands[i].id;
        }
        return cands[cutoff - 1].id;
    };
}

auto compose_prompt(const ConsoleState& state, const std::string& user_text)
    -> std::string {
    std::string out;
    if (state.have_system && !state.system_prompt.empty()) {
        out += state.system_prompt;
        out += "\n";
    }
    for (std::size_t i = 0; i < state.turns.size(); ++i) {
        out += state.turns[i];
        out += "\n";
    }
    out += user_text;
    return out;
}

auto trim(std::string s) -> std::string {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

void print_help(const ConsoleState& state, const LlamaEngine& engine) {
    const auto& cfg = engine.config();
    std::cout << "Commands:\n"
              << "  /help               Show commands + current state.\n"
              << "  /temp <float>       Sampling temperature (0 = greedy). Current: " << state.temperature << "\n"
              << "  /top_p <float>      Nucleus top-p threshold. Current: " << state.top_p << "\n"
              << "  /max <int>          Max generated tokens per response. Current: " << state.max_new_tokens << "\n"
              << "  /system <text>      Set/replace the system prompt prefix.\n"
              << "  /clear              Clear conversation history (keeps system).\n"
              << "  /history            Show per-turn stats for the last response.\n"
              << "  /model              Show model info (path / config / device).\n"
              << "  /save <file>        Persist conversation history to file.\n"
              << "  /load <file>        Load conversation history from file.\n"
              << "  /exit, /quit        Exit.\n\n"
              << "Model path: "    << cfg.model_path     << "\n"
              << "Tokenizer:  "    << cfg.tokenizer_path << "\n"
              << "Device:     "    << Device(cfg.device).name() << "\n"
              << "Synthetic:  "    << (cfg.use_synthetic ? "yes" : "no") << "\n"
              << "Vocab:      "    << engine.tokenizer().vocab_size() << " tokens\n";
}

void print_model_info(const LlamaEngine& engine) {
    const auto& cfg = engine.config().model_config;
    std::cout << "Backend:  " << Device(engine.config().device).name() << "\n"
              << "Mode:     " << (engine.config().use_synthetic ? "Synthetic" : "Real model") << "\n"
              << "Vocab:    " << engine.tokenizer().vocab_size() << " tokens\n"
              << "Hidden:   " << cfg.hidden_size      << "\n"
              << "Layers:   " << cfg.num_layers        << "\n"
              << "Heads:    " << cfg.num_heads         << " (kv=" << cfg.num_kv_heads << ")\n"
              << "HeadDim:  " << cfg.head_dim()        << "\n"
              << "Intermed: " << cfg.intermediate_size << "\n"
              << "Rope θ:   " << cfg.rope_theta        << "\n"
              << "Tied:     " << (cfg.tie_word_embeddings ? "yes" : "no") << "\n";
}

auto save_conversation(const ConsoleState& state, const std::string& path) -> bool {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[save] cannot open " << path << " for writing\n";
        return false;
    }
    f << "# aito conversation snapshot\n";
    if (state.have_system) {
        f << "SYSTEM\n" << state.system_prompt << "\n";
    }
    for (std::size_t i = 0; i < state.turns.size(); ++i) {
        f << ((i % 2 == 0) ? "USER\n" : "ASSISTANT\n")
          << state.turns[i] << "\n";
    }
    return f.good();
}

auto load_conversation(ConsoleState& state, const std::string& path) -> bool {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[load] cannot open " << path << " for reading\n";
        return false;
    }
    state.turns.clear();
    state.have_system = false;
    state.system_prompt.clear();
    std::string line, role, body;
    auto flush = [&]() {
        if (role.empty()) return;
        if (role == "SYSTEM") {
            state.have_system = true;
            state.system_prompt = body;
        } else if (role == "USER" || role == "ASSISTANT") {
            state.turns.push_back(body);
        }
        role.clear();
        body.clear();
    };
    while (std::getline(f, line)) {

        if (!line.empty() && line[0] == '#') {
            flush();
            continue;
        }
        if (line == "SYSTEM" || line == "USER" || line == "ASSISTANT") {
            flush();
            role = line;
        } else {
            if (!body.empty()) body += "\n";
            body += line;
        }
    }
    flush();

    return !f.bad();
}

// 交互控制台指令处理状态
enum class ConsoleAction {
    Handled,
    Exit,
    Passthrough
};

// 处理控制台斜杠指令
auto dispatch_console_command(
    const std::string& cmd,
    ConsoleState& state,
    LlamaEngine& engine
) -> ConsoleAction {
    if (cmd == "/exit" || cmd == "/quit") {
        std::cout << "Exiting...\n";
        return ConsoleAction::Exit;
    }
    if (cmd == "/help") {
        print_help(state, engine);
        std::cout << "\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/model") {
        print_model_info(engine);
        std::cout << "\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/history") {
        std::cout << "Last response: " << state.last_tokens << " tokens in "
                  << std::fixed << std::setprecision(2) << state.last_seconds
                  << "s (" << std::setprecision(1) << state.last_tps << " tok/s)\n"
                  << "Last text: " << state.last_text << "\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/clear") {
        state.turns.clear();
        std::cout << "Conversation cleared (system prompt retained).\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/temp ", 0) == 0) {
        try {
            state.temperature = std::stof(cmd.substr(6));
            std::cout << "Temperature set to " << state.temperature << "\n\n";
        } catch (...) { std::cout << "Invalid temperature value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/top_p ", 0) == 0) {
        try {
            state.top_p = std::stof(cmd.substr(7));
            std::cout << "Top-P set to " << state.top_p << "\n\n";
        } catch (...) { std::cout << "Invalid top-p value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/max ", 0) == 0) {
        try {
            state.max_new_tokens = static_cast<std::size_t>(std::stoul(cmd.substr(5)));
            std::cout << "Max new tokens set to " << state.max_new_tokens << "\n\n";
        } catch (...) { std::cout << "Invalid max tokens value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/system ", 0) == 0) {
        state.system_prompt = cmd.substr(8);
        state.have_system = !state.system_prompt.empty();
        std::cout << "System prompt set (" << state.system_prompt.size()
                  << " chars).\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/system") {
        state.system_prompt.clear();
        state.have_system = false;
        std::cout << "System prompt cleared.\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/save ", 0) == 0) {
        std::string path = trim(cmd.substr(6));
        if (save_conversation(state, path)) {
            std::cout << "Saved to " << path << "\n\n";
        } else {
            std::cout << "Save failed.\n\n";
        }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/load ", 0) == 0) {
        std::string path = trim(cmd.substr(6));
        if (load_conversation(state, path)) {
            std::cout << "Loaded from " << path << " ("
                      << state.turns.size() << " turns)\n\n";
        } else {
            std::cout << "Load failed.\n\n";
        }
        return ConsoleAction::Handled;
    }
    return ConsoleAction::Passthrough;
}

// 执行单轮交互生成并记录历史
auto execute_generation_turn(
    LlamaEngine& engine,
    ConsoleState& state,
    const std::string& cmd
) -> void {
    try {
        std::string composed = compose_prompt(state, cmd);
        auto sampler = make_default_sampler(state);
        auto res = engine.generate(
            composed,
            sampler,
            state.max_new_tokens,
            state.seed++,
            [](std::int32_t , const std::string& piece) {
                std::cout << piece << std::flush;
            }
        );
        std::cout << "\n";
        std::cout << "[" << res.generated_tokens.size() << " tokens in "
                  << std::fixed << std::setprecision(2) << res.elapsed_seconds
                  << "s (" << std::setprecision(1) << res.tokens_per_second
                  << " tok/s)]\n\n";

        state.last_tokens  = res.generated_tokens.size();
        state.last_seconds = res.elapsed_seconds;
        state.last_tps     = res.tokens_per_second;
        state.last_text    = res.text;

        state.turns.push_back(cmd);
        state.turns.push_back(res.text);
    } catch (const std::exception& ex) {
        std::cerr << "\n[Error during generation: " << ex.what() << "]\n\n";
    }
}

}

void run_interactive_console(
    LlamaEngine&      engine,
    std::size_t       max_new_tokens,
    std::uint32_t     seed
) {
    ConsoleState state;
    state.max_new_tokens = max_new_tokens;
    state.seed           = seed;

    const auto& dev_cfg = engine.config().device;
    std::cout << "\n"
              << "===================================================================\n"
              << " Backend:  " << Device(dev_cfg).name() << "\n"
              << " Mode:     " << (engine.config().use_synthetic ? "Synthetic" : "Full Model") << "\n"
              << " Vocab:    " << engine.tokenizer().vocab_size() << " tokens\n"
              << " Commands: /help /temp /top_p /max /system /clear /history\n"
              << "           /model /save /load /exit\n"
              << "===================================================================\n\n";

    std::string line;
    while (true) {
        std::cout << ">>> " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\nExiting...\n";
            break;
        }
        std::string cmd = trim(line);
        if (cmd.empty()) continue;

        const auto action = dispatch_console_command(cmd, state, engine);
        if (action == ConsoleAction::Exit) {
            break;
        }
        if (action == ConsoleAction::Handled) {
            continue;
        }

        execute_generation_turn(engine, state, cmd);
    }
}

auto make_top_p_sampler(float temperature, float top_p, std::mt19937& rng)
    -> std::function<std::int32_t(std::span<const float>)> {
    return [temperature, top_p, &rng](std::span<const float> logits)
        -> std::int32_t {
        if (logits.empty()) return -1;
        if (temperature <= 0.0f || top_p <= 0.0f) {
            std::int32_t best = -1;
            float bv = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < logits.size(); ++i) {
                float v = logits[i];
                if (std::isnan(v)) continue;
                if (best == -1 || v > bv) { bv = v; best = static_cast<std::int32_t>(i); }
            }
            return best;
        }
        float max_v = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (float v : logits) {
            if (std::isfinite(v) && (!any || v > max_v)) { max_v = v; any = true; }
        }
        if (!any) return -1;
        const float inv_t = 1.0f / temperature;
        struct Prob { float p; std::int32_t id; };
        std::vector<Prob> cands;
        cands.reserve(logits.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            float v = logits[i];
            if (std::isnan(v)) continue;
            float p = std::exp((v - max_v) * inv_t);
            sum += p;
            cands.push_back({p, static_cast<std::int32_t>(i)});
        }
        if (cands.empty()) return -1;
        const float inv_sum = static_cast<float>(1.0 / sum);
        for (auto& c : cands) c.p *= inv_sum;
        std::sort(cands.begin(), cands.end(),
                  [](const Prob& a, const Prob& b) { return a.p > b.p; });
        float cum = 0.0f;
        std::size_t cutoff = 0;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].p;
            cutoff = i + 1;
            if (cum >= top_p) break;
        }
        if (cutoff == 0 && !cands.empty()) {
            cutoff = 1;
            cum = cands[0].p;
        }
        std::uniform_real_distribution<float> dist(0.0f, cum);
        float r = dist(rng);
        float acc = 0.0f;
        for (std::size_t i = 0; i < cutoff; ++i) {
            acc += cands[i].p;
            if (r <= acc) return cands[i].id;
        }
        return cands[cutoff - 1].id;
    };
}

}
