#include "llama_graph.h"

#include <cstddef>
#include <stdexcept>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

namespace velomind::examples::llama {

namespace {

inline auto add_seq(velomind::Graph& g,
                    velomind::Tensor  a,
                    velomind::Tensor  b) -> velomind::Tensor {
    return g.op(velomind::Op::Add, a, b);
}

inline auto swiglu(velomind::Graph& g,
                   velomind::Tensor  gate,
                   velomind::Tensor  up) -> velomind::Tensor {
    auto sig_g   = g.op(velomind::Op::Sigmoid, gate);
    auto silu_g  = g.op(velomind::Op::Mul, sig_g, gate);
    return g.op(velomind::Op::Mul, silu_g, up);
}

inline auto apply_linear(velomind::Graph&         g,
                         velomind::Tensor         x,
                         velomind::Tensor         w,
                         velomind::Tensor         scale,
                         velomind::QuantType      quant_type,
                         int                      block_size) -> velomind::Tensor {
    if (quant_type != velomind::QuantType::None && scale) {
        return g.op(
            velomind::OpDescriptor{
                velomind::Op::QuantizedMatMul,
                velomind::QuantizedMatMulAttrs{.quant_type = quant_type, .block_size = block_size}},
            x, w, scale);
    }
    return g.op(velomind::Op::MatMul, x, w);
}

struct QKVTensors {
    velomind::Tensor q3t;
    velomind::Tensor k3t;
    velomind::Tensor v3t;
};

// 投影并重排 Q、K、V 张量并执行 RoPE 旋转位置编码
inline auto project_qkv(
    velomind::Graph& g,
    velomind::Tensor hn,
    std::size_t seq,
    std::size_t H,
    std::size_t H_kv,
    std::size_t D,
    std::size_t layer,
    const ForwardWeights& w
) -> QKVTensors {
    auto q = apply_linear(g, hn, w.wq[layer], layer < w.wq_scale.size() ? w.wq_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
    auto k = apply_linear(g, hn, w.wk[layer], layer < w.wk_scale.size() ? w.wk_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
    auto v = apply_linear(g, hn, w.wv[layer], layer < w.wv_scale.size() ? w.wv_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);

    const shape_t q3_shape  = {static_cast<dim_t>(seq), static_cast<dim_t>(H), static_cast<dim_t>(D)};
    const shape_t kv3_shape = {static_cast<dim_t>(seq), static_cast<dim_t>(H_kv), static_cast<dim_t>(D)};
    auto q3 = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{q3_shape}}, q);
    auto k3 = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{kv3_shape}}, k);
    auto v3 = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{kv3_shape}}, v);

    auto q3t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0, 2}}}, q3);
    auto k3t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0, 2}}}, k3);
    auto v3t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0, 2}}}, v3);

    if (w.cos_cache && w.sin_cache) {
        q3t = g.op(Op::RotaryEmbedding, q3t, w.cos_cache, w.sin_cache);
        k3t = g.op(Op::RotaryEmbedding, k3t, w.cos_cache, w.sin_cache);
    }
    return QKVTensors{.q3t = q3t, .k3t = k3t, .v3t = v3t};
}

// 计算多头注意力上下文张量（支持融合注意力与经典分步计算）
inline auto build_attention_context(
    velomind::Graph& g,
    velomind::Tensor q3t,
    velomind::Tensor k_att,
    velomind::Tensor v_att,
    const ForwardWeights& w
) -> velomind::Tensor {
    if (w.use_fused_attention) {
        return g.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q3t, k_att, v_att);
    }
    auto k_att_T = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{0, 2, 1}}}, k_att);
    auto scores  = g.op(Op::MatMul, q3t, k_att_T);
    if (w.causal_mask) {
        scores = g.op(Op::Add, scores, w.causal_mask);
    }
    auto weights = g.op(Op::Softmax, scores);
    return g.op(Op::MatMul, weights, v_att);
}

// 将注意力上下文张量投影回隐层维度
inline auto project_attention_output(
    velomind::Graph& g,
    velomind::Tensor context,
    std::size_t seq,
    std::size_t hidden,
    std::size_t layer,
    const ForwardWeights& w
) -> velomind::Tensor {
    auto context_t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0, 2}}}, context);
    const shape_t context_2d_shape = {static_cast<dim_t>(seq), static_cast<dim_t>(hidden)};
    auto context_2d = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{context_2d_shape}}, context_t);
    return apply_linear(g, context_2d, w.wo[layer], layer < w.wo_scale.size() ? w.wo_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
}

// 构建 SwiGLU 前馈网络块并叠加残差
inline auto build_mlp_block(
    velomind::Graph& g,
    velomind::Tensor h,
    std::size_t layer,
    const Config& cfg,
    const ForwardWeights& w
) -> velomind::Tensor {
    auto hn2 = g.op(
        OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
        h, w.w_ffn_norm[layer]);
    auto gate = apply_linear(g, hn2, w.w_gate[layer], layer < w.w_gate_scale.size() ? w.w_gate_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
    auto up   = apply_linear(g, hn2, w.w_up[layer], layer < w.w_up_scale.size() ? w.w_up_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
    auto ffn  = swiglu(g, gate, up);

    auto ffn_out = apply_linear(g, ffn, w.w_down[layer], layer < w.w_down_scale.size() ? w.w_down_scale[layer] : Tensor{}, w.quant_type, w.quant_block_size);
    return add_seq(g, h, ffn_out);
}

} // namespace

auto slice_last_token(Graph& g, Tensor t) -> Tensor {
    const auto& shape = t.shape();
    if (shape.empty()) {
        throw std::invalid_argument("slice_last_token: empty shape");
    }
    const int seq = static_cast<int>(shape[0]);
    if (seq <= 1) {
        return t;
    }
    std::vector<int> begins = {seq - 1};
    std::vector<int> ends   = {seq};
    std::vector<int> strides = {1};
    for (std::size_t i = 1; i < shape.size(); ++i) {
        begins.push_back(0);
        ends.push_back(static_cast<int>(shape[i]));
        strides.push_back(1);
    }
    return g.op(OpDescriptor{Op::Slice, SliceAttrs{std::move(begins), std::move(ends), std::move(strides)}}, t);
}

auto build_forward_graph_prefill(
    Graph&                g,
    const Config&         cfg,
    Tensor                tokens,
    const ForwardWeights& w,
    KVCacheHandles*       kv_out,
    bool                  last_token_logits_only
) -> Tensor {
    using Op = velomind::Op;

    if (cfg.num_heads == 0 || cfg.hidden_size % cfg.num_heads != 0) {
        throw std::runtime_error("llama prefill: num_heads must be > 0 and divide hidden_size");
    }
    const std::size_t H       = cfg.num_heads;
    const std::size_t H_kv    = cfg.effective_num_kv_heads();
    const std::size_t D       = cfg.head_dim();
    const std::size_t hidden  = cfg.hidden_size;

    if (H_kv == 0 || H % H_kv != 0) {
        throw std::runtime_error("llama prefill: num_heads must be divisible by num_kv_heads");
    }
    const std::size_t G = H / H_kv;

    if (tokens.shape().size() != 1) {
        throw std::runtime_error("llama prefill: `tokens` must be a 1-D int32 tensor");
    }
    const std::size_t seq = static_cast<std::size_t>(tokens.shape()[0]);

    const std::size_t estimated_nodes = cfg.num_layers * 28 + 10;
    const std::size_t estimated_tensors = estimated_nodes + cfg.num_layers * 12 + 20;
    g.reserve(estimated_tensors, estimated_nodes);

    if (kv_out) {
        kv_out->k.clear();
        kv_out->v.clear();
        kv_out->k.reserve(cfg.num_layers);
        kv_out->v.reserve(cfg.num_layers);
    }

    auto h = g.op(Op::Embedding, w.w_embed, tokens);

    for (std::size_t layer = 0; layer < cfg.num_layers; ++layer) {
        auto hn = g.op(
            OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
            h, w.w_attn_norm[layer]);

        auto [q3t, k3t, v3t] = project_qkv(g, hn, seq, H, H_kv, D, layer, w);

        if (kv_out) {
            kv_out->k.push_back(k3t);
            kv_out->v.push_back(v3t);
            g.mark_output(k3t);
            g.mark_output(v3t);
        }

        auto k_att = (G > 1)
            ? g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{static_cast<int>(G), 0}}, k3t)
            : k3t;
        auto v_att = (G > 1)
            ? g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{static_cast<int>(G), 0}}, v3t)
            : v3t;

        auto context = build_attention_context(g, q3t, k_att, v_att, w);
        auto o = project_attention_output(g, context, seq, hidden, layer, w);
        h = add_seq(g, h, o);

        h = build_mlp_block(g, h, layer, cfg, w);
    }

    auto hn = g.op(
        OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
        h, w.w_final_norm
    );

    if (!w.w_lm_head) {
        throw std::runtime_error("llama prefill: w_lm_head is required");
    }
    auto head_in = last_token_logits_only ? slice_last_token(g, hn) : hn;
    auto logits = apply_linear(g, head_in, w.w_lm_head, w.w_lm_head_scale, w.quant_type, w.quant_block_size);
    if (kv_out) {
        g.mark_output(logits);
    }
    return logits;
}

auto build_forward_graph_decode(
    Graph&                 g,
    const Config&          cfg,
    Tensor                 token,
    const ForwardWeights&  w,
    const KVCacheHandles&  kv_in,
    KVCacheHandles*        kv_out
) -> Tensor {
    using Op = velomind::Op;

    if (cfg.num_heads == 0 || cfg.hidden_size % cfg.num_heads != 0) {
        throw std::runtime_error("llama decode: num_heads must be > 0 and divide hidden_size");
    }
    const std::size_t H      = cfg.num_heads;
    const std::size_t H_kv   = cfg.effective_num_kv_heads();
    const std::size_t D      = cfg.head_dim();
    const std::size_t hidden = cfg.hidden_size;

    if (H_kv == 0 || H % H_kv != 0) {
        throw std::runtime_error("llama decode: num_heads must be divisible by num_kv_heads");
    }
    const std::size_t G = H / H_kv;

    if (token.shape().size() != 1 || token.shape()[0] != 1) {
        throw std::runtime_error("llama decode: `token` must be a 1-D int32 tensor of length 1");
    }
    if (kv_in.k.size() != cfg.num_layers || kv_in.v.size() != cfg.num_layers) {
        throw std::runtime_error("llama decode: kv_in size must match num_layers");
    }

    const std::size_t estimated_nodes = cfg.num_layers * 30 + 10;
    const std::size_t estimated_tensors = estimated_nodes + cfg.num_layers * 14 + 20;
    g.reserve(estimated_tensors, estimated_nodes);

    if (kv_out) {
        kv_out->k.clear();
        kv_out->v.clear();
        kv_out->k.reserve(cfg.num_layers);
        kv_out->v.reserve(cfg.num_layers);
    }

    auto h = g.op(Op::Embedding, w.w_embed, token);

    for (std::size_t layer = 0; layer < cfg.num_layers; ++layer) {
        auto hn = g.op(
            OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
            h, w.w_attn_norm[layer]);

        auto [q3t, k3t, v3t] = project_qkv(g, hn, 1, H, H_kv, D, layer, w);

        auto k_all = g.op(OpDescriptor{Op::Concat, ConcatAttrs{1}}, kv_in.k[layer], k3t);
        auto v_all = g.op(OpDescriptor{Op::Concat, ConcatAttrs{1}}, kv_in.v[layer], v3t);

        if (kv_out) {
            kv_out->k.push_back(k_all);
            kv_out->v.push_back(v_all);
            g.mark_output(k_all);
            g.mark_output(v_all);
        }

        auto k_att = (G > 1)
            ? g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{static_cast<int>(G), 0}}, k_all)
            : k_all;
        auto v_att = (G > 1)
            ? g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{static_cast<int>(G), 0}}, v_all)
            : v_all;

        auto context = build_attention_context(g, q3t, k_att, v_att, w);
        auto o = project_attention_output(g, context, 1, hidden, layer, w);
        h = add_seq(g, h, o);

        h = build_mlp_block(g, h, layer, cfg, w);
    }

    auto hn = g.op(
        OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
        h, w.w_final_norm
    );

    if (!w.w_lm_head) {
        throw std::runtime_error("llama decode: w_lm_head is required");
    }
    auto logits = apply_linear(g, hn, w.w_lm_head, w.w_lm_head_scale, w.quant_type, w.quant_block_size);
    if (kv_out) {
        g.mark_output(logits);
    }
    return logits;
}

auto quantize_forward_weights(Graph&                g,
                              const ForwardWeights& fp_weights,
                              QuantType             type,
                              int                   block_size,
                              DeviceType            device) -> ForwardWeights
{
    if (type == QuantType::None) {
        return fp_weights;
    }

    ForwardWeights qw = fp_weights;
    qw.graph = &g;
    qw.quant_type = type;
    qw.quant_block_size = block_size;

    auto adopt = [&](Tensor t) -> Tensor {
        if (!t) return Tensor{};
        auto stor = t.shared_storage();
        if (!stor) return Tensor{};
        return g.input(stor);
    };

    qw.w_embed      = adopt(fp_weights.w_embed);
    qw.w_final_norm = adopt(fp_weights.w_final_norm);
    qw.cos_cache    = adopt(fp_weights.cos_cache);
    qw.sin_cache    = adopt(fp_weights.sin_cache);
    qw.causal_mask  = adopt(fp_weights.causal_mask);

    const std::size_t num_layers = fp_weights.wq.size();
    qw.w_attn_norm.resize(num_layers);
    qw.w_ffn_norm.resize(num_layers);
    for (std::size_t l = 0; l < num_layers; ++l) {
        qw.w_attn_norm[l] = adopt(fp_weights.w_attn_norm[l]);
        qw.w_ffn_norm[l]  = adopt(fp_weights.w_ffn_norm[l]);
    }

    auto quantize_and_bind = [&](Tensor fp_t, Tensor& out_q, Tensor& out_s) {
        if (!fp_t) return;
        auto stor = fp_t.shared_storage();
        if (!stor) throw std::runtime_error("quantize_forward_weights: tensor storage is null");

        std::shared_ptr<TensorStorage> q_stor;
        std::shared_ptr<TensorStorage> s_stor;

        if (type == QuantType::Int4) {
            auto pair = quantize_storage_int4(*stor, block_size > 0 ? static_cast<std::size_t>(block_size) : 32, device);
            q_stor = pair.first;
            s_stor = pair.second;
        } else {
            auto pair = quantize_storage_int8(*stor, device);
            q_stor = pair.first;
            s_stor = pair.second;
        }

        out_q = g.input(q_stor);
        out_s = g.input(s_stor);
    };

    qw.wq.resize(num_layers);
    qw.wk.resize(num_layers);
    qw.wv.resize(num_layers);
    qw.wo.resize(num_layers);
    qw.w_gate.resize(num_layers);
    qw.w_up.resize(num_layers);
    qw.w_down.resize(num_layers);

    qw.wq_scale.resize(num_layers);
    qw.wk_scale.resize(num_layers);
    qw.wv_scale.resize(num_layers);
    qw.wo_scale.resize(num_layers);
    qw.w_gate_scale.resize(num_layers);
    qw.w_up_scale.resize(num_layers);
    qw.w_down_scale.resize(num_layers);

    for (std::size_t l = 0; l < num_layers; ++l) {
        quantize_and_bind(fp_weights.wq[l], qw.wq[l], qw.wq_scale[l]);
        quantize_and_bind(fp_weights.wk[l], qw.wk[l], qw.wk_scale[l]);
        quantize_and_bind(fp_weights.wv[l], qw.wv[l], qw.wv_scale[l]);
        quantize_and_bind(fp_weights.wo[l], qw.wo[l], qw.wo_scale[l]);
        quantize_and_bind(fp_weights.w_gate[l], qw.w_gate[l], qw.w_gate_scale[l]);
        quantize_and_bind(fp_weights.w_up[l], qw.w_up[l], qw.w_up_scale[l]);
        quantize_and_bind(fp_weights.w_down[l], qw.w_down[l], qw.w_down_scale[l]);
    }

    if (fp_weights.w_lm_head) {
        quantize_and_bind(fp_weights.w_lm_head, qw.w_lm_head, qw.w_lm_head_scale);
    }

    return qw;
}

}
