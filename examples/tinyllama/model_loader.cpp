#include "model_loader.h"

#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

#include "llama_rope_mask.h"

namespace velomind::examples::tinyllama {

auto bind_tinyllama_weights(Graph&        g,
                            const velomind::examples::llama::Config& cfg,
                            std::size_t   seq_len)
    -> velomind::examples::llama::ForwardWeights {
    using velomind::examples::llama::ForwardWeights;
    ForwardWeights w;
    w.graph = &g;

    w.w_embed   = g.input({static_cast<dim_t>(cfg.vocab_size), static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32);
    w.w_lm_head = g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.vocab_size)}, DataType::Float32);
    w.w_final_norm = g.input({static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32);

    w.w_attn_norm.resize(cfg.num_layers);
    w.w_ffn_norm.resize(cfg.num_layers);
    w.wq.resize(cfg.num_layers);
    w.wk.resize(cfg.num_layers);
    w.wv.resize(cfg.num_layers);
    w.wo.resize(cfg.num_layers);
    w.w_gate.resize(cfg.num_layers);
    w.w_up.resize(cfg.num_layers);
    w.w_down.resize(cfg.num_layers);

    const dim_t H_dim  = static_cast<dim_t>(cfg.hidden_size);
    const dim_t I_dim  = static_cast<dim_t>(cfg.intermediate_size);
    const dim_t KV_dim = static_cast<dim_t>(cfg.effective_num_kv_heads() * cfg.head_dim());

    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        w.w_attn_norm[l] = g.input({H_dim}, DataType::Float32);
        w.w_ffn_norm[l]  = g.input({H_dim}, DataType::Float32);
        w.wq[l]          = g.input({H_dim, H_dim}, DataType::Float32);
        w.wk[l]          = g.input({H_dim, KV_dim}, DataType::Float32);
        w.wv[l]          = g.input({H_dim, KV_dim}, DataType::Float32);
        w.wo[l]          = g.input({H_dim, H_dim}, DataType::Float32);
        w.w_gate[l]      = g.input({H_dim, I_dim}, DataType::Float32);
        w.w_up[l]        = g.input({H_dim, I_dim}, DataType::Float32);
        w.w_down[l]      = g.input({I_dim, H_dim}, DataType::Float32);
    }

    if (seq_len > 0) {
        const dim_t S_dim = static_cast<dim_t>(seq_len);
        const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
        const dim_t NH_dim = static_cast<dim_t>(cfg.num_heads);
        w.cos_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
        w.sin_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
        w.causal_mask = g.input({NH_dim, S_dim, S_dim}, DataType::Float32);
    }

    return w;
}

namespace {

void transpose_2d(const float* src, std::size_t rows, std::size_t cols, float* dst, float scale = 1.0f) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c] * scale;
        }
    }
}

// 加载单个 Transformer 层的规范化、注意力与前馈网络权重
auto populate_layer_weights(
    const SafetensorsFile& sf,
    const velomind::examples::llama::Config& cfg,
    velomind::examples::llama::ForwardWeights& w,
    std::size_t l,
    float scale_q
) -> void {
    auto copy_raw = [](Tensor& dst, const std::shared_ptr<TensorStorage>& src_storage) {
        dst.copy_from_host(std::span<const std::byte>(
            static_cast<const std::byte*>(src_storage->data), src_storage->size_bytes));
    };

    std::string pfx = "model.layers." + std::to_string(l) + ".";

    if (sf.tensors.count(pfx + "input_layernorm.weight")) {
        copy_raw(w.w_attn_norm[l], sf.tensors.at(pfx + "input_layernorm.weight"));
    }
    if (sf.tensors.count(pfx + "post_attention_layernorm.weight")) {
        copy_raw(w.w_ffn_norm[l], sf.tensors.at(pfx + "post_attention_layernorm.weight"));
    }

    if (sf.tensors.count(pfx + "self_attn.q_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "self_attn.q_proj.weight");
        std::vector<float> buf(cfg.hidden_size * cfg.hidden_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.hidden_size, cfg.hidden_size, buf.data(), scale_q);
        w.wq[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    const std::size_t kv_dim = cfg.effective_num_kv_heads() * cfg.head_dim();
    if (sf.tensors.count(pfx + "self_attn.k_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "self_attn.k_proj.weight");
        std::vector<float> buf(cfg.hidden_size * kv_dim);
        transpose_2d(static_cast<const float*>(stor->data),
                     kv_dim, cfg.hidden_size, buf.data());
        w.wk[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    if (sf.tensors.count(pfx + "self_attn.v_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "self_attn.v_proj.weight");
        std::vector<float> buf(cfg.hidden_size * kv_dim);
        transpose_2d(static_cast<const float*>(stor->data),
                     kv_dim, cfg.hidden_size, buf.data());
        w.wv[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    if (sf.tensors.count(pfx + "self_attn.o_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "self_attn.o_proj.weight");
        std::vector<float> buf(cfg.hidden_size * cfg.hidden_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.hidden_size, cfg.hidden_size, buf.data());
        w.wo[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    if (sf.tensors.count(pfx + "mlp.gate_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "mlp.gate_proj.weight");
        std::vector<float> buf(cfg.hidden_size * cfg.intermediate_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.intermediate_size, cfg.hidden_size, buf.data());
        w.w_gate[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
    if (sf.tensors.count(pfx + "mlp.up_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "mlp.up_proj.weight");
        std::vector<float> buf(cfg.hidden_size * cfg.intermediate_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.intermediate_size, cfg.hidden_size, buf.data());
        w.w_up[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
    if (sf.tensors.count(pfx + "mlp.down_proj.weight")) {
        const auto& stor = sf.tensors.at(pfx + "mlp.down_proj.weight");
        std::vector<float> buf(cfg.intermediate_size * cfg.hidden_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.hidden_size, cfg.intermediate_size, buf.data());
        w.w_down[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
}

// 预计算 RoPE 旋转编码与因果掩码缓存
auto populate_positional_caches(
    const velomind::examples::llama::Config& cfg,
    velomind::examples::llama::ForwardWeights& w,
    std::size_t seq_len
) -> void {
    using velomind::examples::llama::compute_rope_cache;
    using velomind::examples::llama::compute_causal_mask;
    if (seq_len > 0 && w.cos_cache && w.sin_cache) {
        std::vector<float> cos_buf, sin_buf;
        compute_rope_cache(seq_len, cfg.head_dim(), cfg.rope_theta, cos_buf, sin_buf);
        w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
        w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));
    }
    if (seq_len > 0 && w.causal_mask) {
        std::vector<float> mask_buf;
        compute_causal_mask(cfg.num_heads, seq_len, mask_buf);
        w.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));
    }
}

}

void populate_tinyllama_weights(
    const SafetensorsFile& sf,
    const velomind::examples::llama::Config& cfg,
    velomind::examples::llama::ForwardWeights& w,
    std::size_t seq_len
) {
    auto copy_raw = [](Tensor& dst, const std::shared_ptr<TensorStorage>& src_storage) {
        dst.copy_from_host(std::span<const std::byte>(
            static_cast<const std::byte*>(src_storage->data), src_storage->size_bytes));
    };

    if (sf.tensors.count("model.embed_tokens.weight")) {
        copy_raw(w.w_embed, sf.tensors.at("model.embed_tokens.weight"));
    }

    if (sf.tensors.count("model.norm.weight")) {
        copy_raw(w.w_final_norm, sf.tensors.at("model.norm.weight"));
    }

    if (sf.tensors.count("lm_head.weight")) {
        const auto& stor = sf.tensors.at("lm_head.weight");
        std::vector<float> buf(cfg.hidden_size * cfg.vocab_size);
        transpose_2d(static_cast<const float*>(stor->data), cfg.vocab_size, cfg.hidden_size, buf.data());
        w.w_lm_head.copy_from_host(std::as_bytes(std::span(buf)));
    }

    // 将 1/sqrt(head_dim) 预乘进 Q 权重，省去运行时标量乘法。
    const float scale_q = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim()));
    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        populate_layer_weights(sf, cfg, w, l, scale_q);
    }

    populate_positional_caches(cfg, w, seq_len);
}

}
