#include "model.h"

#include "velomind/safetensors.h"
#include "velomind/types.h"

#include "llama_graph.h"
#include "llama_rope_mask.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace velomind::examples::smollm2 {

namespace {

    inline auto bf16_to_f32(std::uint16_t bits) -> float {
        std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
        float out;
        std::memcpy(&out, &f32_bits, sizeof(out));
        return out;
    }

    // 计算后端暂以 FP32 为主，加载时精确扩展 BF16 的高 16 位。
    auto convert_bf16_storage_to_f32(const std::shared_ptr<TensorStorage>& stor,
                                     std::size_t numel) -> std::vector<float> {
        if (stor->dtype != DataType::BFloat16) {
            throw std::runtime_error(
                "smollm2::convert_bf16_storage_to_f32: storage is not BFloat16");
        }
        const auto* src = static_cast<const std::uint16_t*>(stor->data);
        std::vector<float> out(numel);
        for (std::size_t i = 0; i < numel; ++i) {
            out[i] = bf16_to_f32(src[i]);
        }
        return out;
    }

    void transpose_2d(const float* src, std::size_t rows, std::size_t cols,
                      float* dst, float scale = 1.0f) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                dst[c * rows + r] = src[r * cols + c] * scale;
            }
        }
    }

}

using velomind::examples::llama::ForwardWeights;
using velomind::examples::llama::Config;
using velomind::examples::llama::compute_rope_cache;
using velomind::examples::llama::compute_causal_mask;

auto bind_smollm2_weights(Graph&        g,
                          const Config& cfg,
                          std::size_t   seq_len) -> ForwardWeights {
    ForwardWeights w;
    w.graph = &g;

    w.w_embed      = g.input({static_cast<dim_t>(cfg.vocab_size),    static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32);
    w.w_lm_head    = g.input({static_cast<dim_t>(cfg.hidden_size),   static_cast<dim_t>(cfg.vocab_size)}, DataType::Float32);
    w.w_final_norm = g.input({static_cast<dim_t>(cfg.hidden_size)},   DataType::Float32);

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
        const dim_t S_dim  = static_cast<dim_t>(seq_len);
        const dim_t HD_dim = static_cast<dim_t>(cfg.head_dim() / 2);
        const dim_t NH_dim = static_cast<dim_t>(cfg.num_heads);
        w.cos_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
        w.sin_cache   = g.input({S_dim, HD_dim}, DataType::Float32);
        w.causal_mask = g.input({NH_dim, S_dim, S_dim}, DataType::Float32);
    }

    return w;
}

namespace {

// 从 safetensors 中提取必须存在的 BF16 张量并转换为 FP32
auto require_bf16_f32_tensor(
    const SafetensorsFile& sf,
    const std::string& name,
    std::size_t numel
) -> std::vector<float> {
    if (!sf.tensors.count(name)) {
        throw std::runtime_error(
            std::string("smollm2: required tensor '") + name + "' not found in safetensors"
        );
    }
    return convert_bf16_storage_to_f32(sf.tensors.at(name), numel);
}

// 加载或绑定语言模型输出头权重 (lm_head)
auto populate_lm_head_weights(
    const SafetensorsFile& sf,
    const Config& cfg,
    ForwardWeights& w,
    const std::vector<float>& embed
) -> void {
    if (sf.tensors.count("lm_head.weight")) {
        const auto& stor = sf.tensors.at("lm_head.weight");
        if (stor->dtype == DataType::Float32) {
            std::vector<float> buf(cfg.hidden_size * cfg.vocab_size);
            transpose_2d(static_cast<const float*>(stor->data),
                         cfg.vocab_size, cfg.hidden_size, buf.data());
            w.w_lm_head.copy_from_host(std::as_bytes(std::span(buf)));
        } else if (stor->dtype == DataType::BFloat16) {
            auto raw = convert_bf16_storage_to_f32(
                stor, cfg.vocab_size * cfg.hidden_size
            );
            std::vector<float> buf(cfg.hidden_size * cfg.vocab_size);
            transpose_2d(raw.data(), cfg.vocab_size, cfg.hidden_size, buf.data());
            w.w_lm_head.copy_from_host(std::as_bytes(std::span(buf)));
        } else {
            throw std::runtime_error(
                "smollm2: lm_head.weight has unsupported dtype"
            );
        }
    } else if (cfg.tie_word_embeddings) {
        std::vector<float> buf(cfg.hidden_size * cfg.vocab_size);
        transpose_2d(embed.data(), cfg.vocab_size, cfg.hidden_size, buf.data());
        w.w_lm_head.copy_from_host(std::as_bytes(std::span(buf)));
    } else {
        throw std::runtime_error(
            "smollm2: lm_head.weight absent and tie_word_embeddings is false"
        );
    }
}

// 加载单个 Transformer 层的规范化、注意力与前馈网络权重
auto populate_layer_weights(
    const SafetensorsFile& sf,
    const Config& cfg,
    ForwardWeights& w,
    std::size_t l,
    float scale_q
) -> void {
    std::string pfx = "model.layers." + std::to_string(l) + ".";

    auto an = require_bf16_f32_tensor(sf, pfx + "input_layernorm.weight", cfg.hidden_size);
    w.w_attn_norm[l].copy_from_host(std::as_bytes(std::span(an)));
    auto fn = require_bf16_f32_tensor(sf, pfx + "post_attention_layernorm.weight", cfg.hidden_size);
    w.w_ffn_norm[l].copy_from_host(std::as_bytes(std::span(fn)));

    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "self_attn.q_proj.weight", cfg.hidden_size * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * cfg.hidden_size);
        transpose_2d(raw.data(), cfg.hidden_size, cfg.hidden_size, buf.data(), scale_q);
        w.wq[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    const std::size_t kv_dim = cfg.effective_num_kv_heads() * cfg.head_dim();
    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "self_attn.k_proj.weight", kv_dim * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * kv_dim);
        transpose_2d(raw.data(), kv_dim, cfg.hidden_size, buf.data());
        w.wk[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "self_attn.v_proj.weight", kv_dim * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * kv_dim);
        transpose_2d(raw.data(), kv_dim, cfg.hidden_size, buf.data());
        w.wv[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "self_attn.o_proj.weight", cfg.hidden_size * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * cfg.hidden_size);
        transpose_2d(raw.data(), cfg.hidden_size, cfg.hidden_size, buf.data());
        w.wo[l].copy_from_host(std::as_bytes(std::span(buf)));
    }

    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "mlp.gate_proj.weight", cfg.intermediate_size * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * cfg.intermediate_size);
        transpose_2d(raw.data(), cfg.intermediate_size, cfg.hidden_size, buf.data());
        w.w_gate[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "mlp.up_proj.weight", cfg.intermediate_size * cfg.hidden_size
        );
        std::vector<float> buf(cfg.hidden_size * cfg.intermediate_size);
        transpose_2d(raw.data(), cfg.intermediate_size, cfg.hidden_size, buf.data());
        w.w_up[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
    {
        auto raw = require_bf16_f32_tensor(
            sf, pfx + "mlp.down_proj.weight", cfg.hidden_size * cfg.intermediate_size
        );
        std::vector<float> buf(cfg.intermediate_size * cfg.hidden_size);
        transpose_2d(raw.data(), cfg.hidden_size, cfg.intermediate_size, buf.data());
        w.w_down[l].copy_from_host(std::as_bytes(std::span(buf)));
    }
}

// 预计算 RoPE 旋转编码与因果掩码缓存
auto populate_positional_caches(
    const Config& cfg,
    ForwardWeights& w,
    std::size_t seq_len
) -> void {
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

} // namespace

void populate_smollm2_weights(
    const SafetensorsFile& sf,
    const Config&          cfg,
    ForwardWeights&        w,
    std::size_t            seq_len
) {
    auto embed = require_bf16_f32_tensor(
        sf, "model.embed_tokens.weight", cfg.vocab_size * cfg.hidden_size
    );
    w.w_embed.copy_from_host(std::as_bytes(std::span(embed)));

    auto final_norm = require_bf16_f32_tensor(
        sf, "model.norm.weight", cfg.hidden_size
    );
    w.w_final_norm.copy_from_host(std::as_bytes(std::span(final_norm)));

    populate_lm_head_weights(sf, cfg, w, embed);

    const float scale_q = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim()));
    for (std::size_t l = 0; l < cfg.num_layers; ++l) {
        populate_layer_weights(sf, cfg, w, l, scale_q);
    }

    populate_positional_caches(cfg, w, seq_len);
}

}
