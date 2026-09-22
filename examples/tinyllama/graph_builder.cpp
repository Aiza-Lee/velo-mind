#include "graph_builder.h"

#include <cstddef>
#include <stdexcept>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

namespace velomind::examples::tinyllama {

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

}

auto build_forward_graph(Graph&            g,
                         const velomind::examples::llama::Config& cfg,
                         Tensor            token,
                         const velomind::examples::llama::ForwardWeights& w) -> Tensor {
    using Op = velomind::Op;

    if (cfg.num_layers == 0) {
        throw std::runtime_error("tinyllama v1: num_layers must be > 0");
    }
    if (w.wq.size() != cfg.num_layers ||
        w.wk.size() != cfg.num_layers ||
        w.wv.size() != cfg.num_layers ||
        w.wo.size() != cfg.num_layers ||
        w.w_attn_norm.size() != cfg.num_layers ||
        w.w_ffn_norm.size()  != cfg.num_layers ||
        w.w_gate.size() != cfg.num_layers ||
        w.w_up.size()   != cfg.num_layers ||
        w.w_down.size() != cfg.num_layers) {
        throw std::runtime_error("tinyllama v1: weight vector size mismatch");
    }

    // 依据模型层数与典型拓扑估算张量与算子节点数，提前预留容量以消除构图期间的动态重分配。
    const std::size_t estimated_nodes = cfg.num_layers * 25 + 10;
    const std::size_t estimated_tensors = estimated_nodes + cfg.num_layers * 10 + 20;
    g.reserve(estimated_tensors, estimated_nodes);

    auto h = g.op(Op::Embedding, w.w_embed, token);

    for (std::size_t layer = 0; layer < cfg.num_layers; ++layer) {

        auto hn = g.op(
            OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
            h, w.w_attn_norm[layer]);

        auto q = g.op(Op::MatMul, hn, w.wq[layer]);
        auto k = g.op(Op::MatMul, hn, w.wk[layer]);
        auto v = g.op(Op::MatMul, hn, w.wv[layer]);
        (void)q; (void)k;

        auto o = g.op(Op::MatMul, v, w.wo[layer]);
        h = add_seq(g, h, o);

        auto hn2 = g.op(
            OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
            h, w.w_ffn_norm[layer]);
        auto gate = g.op(Op::MatMul, hn2, w.w_gate[layer]);
        auto up   = g.op(Op::MatMul, hn2, w.w_up[layer]);
        auto ffn  = swiglu(g, gate, up);

        auto ffn_out = g.op(Op::MatMul, ffn, w.w_down[layer]);
        h = add_seq(g, h, ffn_out);
    }

    auto hn = g.op(
        OpDescriptor{Op::RMSNorm, RMSNormAttrs{cfg.rms_norm_eps, -1}},
        h, w.w_final_norm);

    if (!w.w_lm_head) {
        throw std::runtime_error(
            "tinyllama v1: w_lm_head is required (v1 doesn't tie to w_embed)");
    }
    return g.op(Op::MatMul, hn, w.w_lm_head);
}

}
