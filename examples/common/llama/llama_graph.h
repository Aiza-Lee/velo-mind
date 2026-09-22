#pragma once

#include <vector>

#include "velomind/graph.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "velomind/quant.h"

#include "llama_config.h"

namespace velomind::examples::llama {

struct ForwardWeights {
    Graph*                       graph  = nullptr;
    Tensor                       w_embed;
    Tensor                       w_lm_head;
    std::vector<Tensor>          w_attn_norm;
    std::vector<Tensor>          wq, wk, wv, wo;
    std::vector<Tensor>          w_ffn_norm;
    std::vector<Tensor>          w_gate, w_up, w_down;
    Tensor                       w_final_norm;
    Tensor                       cos_cache;
    Tensor                       sin_cache;
    Tensor                       causal_mask;
    bool                         use_fused_attention = false;
    QuantType                    quant_type          = QuantType::None;
    int                          quant_block_size    = 0;
    std::vector<Tensor>          wq_scale, wk_scale, wv_scale, wo_scale;
    std::vector<Tensor>          w_gate_scale, w_up_scale, w_down_scale;
    Tensor                       w_lm_head_scale;
};

struct KVCacheHandles {
    std::vector<Tensor> k;
    std::vector<Tensor> v;
};

// 提取最后一行的张量切片 [1, hidden]，专供生成路径避免全序列 LM head 计算。
auto slice_last_token(Graph& g, Tensor t) -> Tensor;

// 构建首阶段全序列预填充计算图；last_token_logits_only 为 true 时仅针对最后一个 token 计算 logits，避免生成路径的全序列 LM head 开销。
auto build_forward_graph_prefill(Graph&                g,
                                 const Config&         cfg,
                                 Tensor                tokens,
                                 const ForwardWeights& w,
                                 KVCacheHandles*       kv_out = nullptr,
                                 bool                  last_token_logits_only = false) -> Tensor;

// 构建自回归单 token 解码计算图；拼接历史 KV 缓存，无需因果掩码，输出最新单 token logits 与更新后的各层 KV。
auto build_forward_graph_decode(Graph&                 g,
                                const Config&          cfg,
                                Tensor                 token,
                                const ForwardWeights&  w,
                                const KVCacheHandles&  kv_in,
                                KVCacheHandles*        kv_out = nullptr) -> Tensor;

// 将一组已加载的浮点权重（FP32/FP16/BF16）量化为 INT8/INT4 并绑定到目标图 g
auto quantize_forward_weights(Graph&                g,
                              const ForwardWeights& fp_weights,
                              QuantType             type,
                              int                   block_size = 0,
                              DeviceType            device = DeviceType::CPU) -> ForwardWeights;

}
