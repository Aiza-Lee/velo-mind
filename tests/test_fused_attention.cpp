#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"
#include "velomind/device.h"
#include "velomind/dtype.h"

#include "test_helpers.h"
#include "test_model_assets.h"
#include "llama_graph.h"
#include "llama_rope_mask.h"

using namespace velomind;
using namespace velomind_test;

namespace {

inline auto make_pattern_data(std::size_t n, float base = 0.1f, float step = 0.05f) -> std::vector<float> {
    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        int pattern = static_cast<int>(i % 17) - 8;
        data[i] = base + static_cast<float>(pattern) * step;
    }
    return data;
}

} // namespace

TEST_CASE("FusedAttention operator validation", "[validation][fused_attention]") {
    Graph g;
    auto q = g.input({2, 4, 16, 64}, DataType::Float32);
    auto k = g.input({2, 4, 16, 64}, DataType::Float32);
    auto v = g.input({2, 4, 16, 64}, DataType::Float32);

    // 输入数量校验
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k, v, q), std::invalid_argument);

    // 秩与维度校验
    auto q_2d = g.input({16, 64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q_2d, q_2d, q_2d), std::invalid_argument);

    auto k_bad_batch = g.input({3, 4, 16, 64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k_bad_batch, v), std::invalid_argument);

    auto k_bad_dim = g.input({2, 4, 16, 32}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k_bad_dim, v), std::invalid_argument);

    auto v_bad_seq = g.input({2, 4, 8, 64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k, v_bad_seq), std::invalid_argument);

    auto k_bad_head = g.input({2, 3, 16, 64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q, k_bad_head, k_bad_head), std::invalid_argument);

    // 因果遮蔽约束：因果模式下要求 S_k >= S_q
    auto q_long = g.input({2, 4, 32, 64}, DataType::Float32);
    auto k_short = g.input({2, 4, 16, 64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
                           q_long, k_short, k_short), std::invalid_argument);
    // 非因果模式下允许 S_k < S_q
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = false}},
                         q_long, k_short, k_short));

    // 数据类型校验
    auto q_i32 = g.input({2, 4, 16, 64}, DataType::Int32);
    REQUIRE_THROWS_AS(g.op(Op::FusedAttention, q_i32, q_i32, q_i32), std::invalid_argument);

    // 正常构建签名
    REQUIRE_NOTHROW(g.op(Op::FusedAttention, q, k, v));
}

TEST_CASE("FusedAttention vs Unfused Attention mathematical equivalence (CPU)", "[cpu][fused_attention]") {
    const std::vector<dim_t> test_seq_lens = {1, 4, 16, 64};
    const dim_t H = 4;
    const dim_t D = 64;

    for (dim_t S : test_seq_lens) {
        DYNAMIC_SECTION("Sequence length S = " << S) {
            // 构建未融合算子图 (Q @ K^T -> Mask -> Softmax -> @ V)
            Graph g_unfused;
            auto q_u = g_unfused.input({H, S, D}, DataType::Float32);
            auto k_u = g_unfused.input({H, S, D}, DataType::Float32);
            auto v_u = g_unfused.input({H, S, D}, DataType::Float32);
            auto mask_u = g_unfused.input({H, S, S}, DataType::Float32);

            auto kt_u = g_unfused.op(OpDescriptor{Op::Transpose, TransposeAttrs{{0, 2, 1}}}, k_u);
            auto scores_u = g_unfused.op(Op::MatMul, q_u, kt_u);
            auto masked_scores_u = g_unfused.op(Op::Add, scores_u, mask_u);
            auto weights_u = g_unfused.op(Op::Softmax, masked_scores_u);
            auto out_u = g_unfused.op(Op::MatMul, weights_u, v_u);

            auto exec_unfused = g_unfused.build(DeviceType::CPU);
            REQUIRE(exec_unfused);

            // 构建融合注意力图
            Graph g_fused;
            auto q_f = g_fused.input({H, S, D}, DataType::Float32);
            auto k_f = g_fused.input({H, S, D}, DataType::Float32);
            auto v_f = g_fused.input({H, S, D}, DataType::Float32);
            auto out_f = g_fused.op(
                OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
                q_f, k_f, v_f);

            auto exec_fused = g_fused.build(DeviceType::CPU);
            REQUIRE(exec_fused);

            const std::size_t numel_qkv = static_cast<std::size_t>(H * S * D);
            auto q_data = make_pattern_data(numel_qkv, 0.1f, 0.02f);
            auto k_data = make_pattern_data(numel_qkv, -0.05f, 0.015f);
            auto v_data = make_pattern_data(numel_qkv, 0.2f, 0.03f);

            // 因果掩码数据 [H, S, S]
            std::vector<float> mask_data(static_cast<std::size_t>(H * S * S), 0.0f);
            for (dim_t h = 0; h < H; ++h) {
                for (dim_t i = 0; i < S; ++i) {
                    for (dim_t j = 0; j < S; ++j) {
                        if (j > i) {
                            mask_data[static_cast<std::size_t>((h * S + i) * S + j)] = -1e9f;
                        }
                    }
                }
            }

            // 执行未融合图
            q_u.copy_from_host(as_bytes(q_data));
            k_u.copy_from_host(as_bytes(k_data));
            v_u.copy_from_host(as_bytes(v_data));
            mask_u.copy_from_host(as_bytes(mask_data));
            exec_unfused->execute();

            std::vector<float> res_unfused(numel_qkv);
            out_u.copy_to_host(as_writeable_bytes(res_unfused));

            // 执行融合注意力图
            q_f.copy_from_host(as_bytes(q_data));
            k_f.copy_from_host(as_bytes(k_data));
            v_f.copy_from_host(as_bytes(v_data));
            exec_fused->execute();

            std::vector<float> res_fused(numel_qkv);
            out_f.copy_to_host(as_writeable_bytes(res_fused));

            require_all_finite(res_unfused, "res_unfused");
            require_all_finite(res_fused, "res_fused");

            // 验证数值一致性 (max diff < 1e-4)
            float max_diff = 0.0f;
            for (std::size_t i = 0; i < numel_qkv; ++i) {
                float diff = std::abs(res_unfused[i] - res_fused[i]);
                if (diff > max_diff) max_diff = diff;
            }
            CHECK(max_diff < 1e-4f);
        }
    }
}

TEST_CASE("FusedAttention non-causal mode (CPU)", "[cpu][fused_attention]") {
    const dim_t H = 2;
    const dim_t S_q = 8;
    const dim_t S_k = 12;
    const dim_t D = 32;

    Graph g_unfused;
    auto q_u = g_unfused.input({H, S_q, D}, DataType::Float32);
    auto k_u = g_unfused.input({H, S_k, D}, DataType::Float32);
    auto v_u = g_unfused.input({H, S_k, D}, DataType::Float32);

    auto kt_u = g_unfused.op(OpDescriptor{Op::Transpose, TransposeAttrs{{0, 2, 1}}}, k_u);
    auto scores_u = g_unfused.op(Op::MatMul, q_u, kt_u);
    auto weights_u = g_unfused.op(Op::Softmax, scores_u);
    auto out_u = g_unfused.op(Op::MatMul, weights_u, v_u);

    auto exec_unfused = g_unfused.build(DeviceType::CPU);
    REQUIRE(exec_unfused);

    Graph g_fused;
    auto q_f = g_fused.input({H, S_q, D}, DataType::Float32);
    auto k_f = g_fused.input({H, S_k, D}, DataType::Float32);
    auto v_f = g_fused.input({H, S_k, D}, DataType::Float32);
    auto out_f = g_fused.op(
        OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = false}},
        q_f, k_f, v_f);

    auto exec_fused = g_fused.build(DeviceType::CPU);
    REQUIRE(exec_fused);

    auto q_data = make_pattern_data(static_cast<std::size_t>(H * S_q * D), 0.1f, 0.03f);
    auto k_data = make_pattern_data(static_cast<std::size_t>(H * S_k * D), -0.1f, 0.02f);
    auto v_data = make_pattern_data(static_cast<std::size_t>(H * S_k * D), 0.05f, 0.01f);

    q_u.copy_from_host(as_bytes(q_data));
    k_u.copy_from_host(as_bytes(k_data));
    v_u.copy_from_host(as_bytes(v_data));
    exec_unfused->execute();

    std::vector<float> res_u(static_cast<std::size_t>(H * S_q * D));
    out_u.copy_to_host(as_writeable_bytes(res_u));

    q_f.copy_from_host(as_bytes(q_data));
    k_f.copy_from_host(as_bytes(k_data));
    v_f.copy_from_host(as_bytes(v_data));
    exec_fused->execute();

    std::vector<float> res_f(static_cast<std::size_t>(H * S_q * D));
    out_f.copy_to_host(as_writeable_bytes(res_f));

    require_all_finite(res_u, "non_causal_unfused");
    require_all_finite(res_f, "non_causal_fused");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < res_u.size(); ++i) {
        float diff = std::abs(res_u[i] - res_f[i]);
        if (diff > max_diff) max_diff = diff;
    }
    CHECK(max_diff < 1e-4f);
}

TEST_CASE("FusedAttention Grouped Query Attention (GQA) (CPU)", "[cpu][fused_attention]") {
    const dim_t H_q = 6;
    const dim_t H_kv = 2;
    const dim_t S = 8;
    const dim_t D = 32;

    // 参考路径：先 RepeatKV 展开至 6 头，再执行融合注意力
    Graph g_ref;
    auto q_r = g_ref.input({H_q, S, D}, DataType::Float32);
    auto k_r = g_ref.input({H_kv, S, D}, DataType::Float32);
    auto v_r = g_ref.input({H_kv, S, D}, DataType::Float32);
    auto k_rep = g_ref.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = 3, .axis = 0}}, k_r);
    auto v_rep = g_ref.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = 3, .axis = 0}}, v_r);
    auto out_r = g_ref.op(
        OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
        q_r, k_rep, v_rep);

    auto exec_ref = g_ref.build(DeviceType::CPU);
    REQUIRE(exec_ref);

    // 原生 GQA 融合路径：直接传入 H_kv 头的 K 与 V，算子内部隐式按头分组展开
    Graph g_gqa;
    auto q_g = g_gqa.input({H_q, S, D}, DataType::Float32);
    auto k_g = g_gqa.input({H_kv, S, D}, DataType::Float32);
    auto v_g = g_gqa.input({H_kv, S, D}, DataType::Float32);
    auto out_g = g_gqa.op(
        OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
        q_g, k_g, v_g);

    auto exec_gqa = g_gqa.build(DeviceType::CPU);
    REQUIRE(exec_gqa);

    auto q_data = make_pattern_data(static_cast<std::size_t>(H_q * S * D), 0.1f, 0.02f);
    auto k_data = make_pattern_data(static_cast<std::size_t>(H_kv * S * D), -0.05f, 0.015f);
    auto v_data = make_pattern_data(static_cast<std::size_t>(H_kv * S * D), 0.2f, 0.03f);

    q_r.copy_from_host(as_bytes(q_data));
    k_r.copy_from_host(as_bytes(k_data));
    v_r.copy_from_host(as_bytes(v_data));
    exec_ref->execute();

    std::vector<float> res_ref(static_cast<std::size_t>(H_q * S * D));
    out_r.copy_to_host(as_writeable_bytes(res_ref));

    q_g.copy_from_host(as_bytes(q_data));
    k_g.copy_from_host(as_bytes(k_data));
    v_g.copy_from_host(as_bytes(v_data));
    exec_gqa->execute();

    std::vector<float> res_gqa(static_cast<std::size_t>(H_q * S * D));
    out_g.copy_to_host(as_writeable_bytes(res_gqa));

    require_all_finite(res_ref, "gqa_ref");
    require_all_finite(res_gqa, "gqa_native");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < res_ref.size(); ++i) {
        float diff = std::abs(res_ref[i] - res_gqa[i]);
        if (diff > max_diff) max_diff = diff;
    }
    CHECK(max_diff < 1e-5f);
}

TEST_CASE("FusedAttention Float16 and BFloat16 precision (CPU)", "[cpu][fused_attention][dtype]") {
    const dim_t H = 2;
    const dim_t S = 8;
    const dim_t D = 32;
    const std::size_t numel = static_cast<std::size_t>(H * S * D);

    auto f32_q = make_pattern_data(numel, 0.2f, 0.03f);
    auto f32_k = make_pattern_data(numel, -0.1f, 0.02f);
    auto f32_v = make_pattern_data(numel, 0.15f, 0.01f);

    // 计算 Float32 黄金结果
    Graph g_f32;
    auto q_32 = g_f32.input({H, S, D}, DataType::Float32);
    auto k_32 = g_f32.input({H, S, D}, DataType::Float32);
    auto v_32 = g_f32.input({H, S, D}, DataType::Float32);
    auto out_32 = g_f32.op(
        OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
        q_32, k_32, v_32);
    auto exec_32 = g_f32.build(DeviceType::CPU);
    q_32.copy_from_host(as_bytes(f32_q));
    k_32.copy_from_host(as_bytes(f32_k));
    v_32.copy_from_host(as_bytes(f32_v));
    exec_32->execute();
    std::vector<float> golden(numel);
    out_32.copy_to_host(as_writeable_bytes(golden));

    // 测试 Float16
    {
        std::vector<float16_t> f16_q(numel), f16_k(numel), f16_v(numel);
        convert_dtype(f32_q.data(), DataType::Float32, f16_q.data(), DataType::Float16, numel);
        convert_dtype(f32_k.data(), DataType::Float32, f16_k.data(), DataType::Float16, numel);
        convert_dtype(f32_v.data(), DataType::Float32, f16_v.data(), DataType::Float16, numel);

        Graph g_f16;
        auto q_16 = g_f16.input({H, S, D}, DataType::Float16);
        auto k_16 = g_f16.input({H, S, D}, DataType::Float16);
        auto v_16 = g_f16.input({H, S, D}, DataType::Float16);
        auto out_16 = g_f16.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q_16, k_16, v_16);
        auto exec_16 = g_f16.build(DeviceType::CPU);
        REQUIRE(exec_16);

        q_16.copy_from_host(as_bytes(f16_q));
        k_16.copy_from_host(as_bytes(f16_k));
        v_16.copy_from_host(as_bytes(f16_v));
        exec_16->execute();

        std::vector<float16_t> res_16(numel);
        out_16.copy_to_host(as_writeable_bytes(res_16));

        std::vector<float> res_f32(numel);
        convert_dtype(res_16.data(), DataType::Float16, res_f32.data(), DataType::Float32, numel);
        require_all_finite(res_f32, "cpu_fused_attn_f16");

        float max_diff = 0.0f;
        for (std::size_t i = 0; i < numel; ++i) {
            float diff = std::abs(golden[i] - res_f32[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(max_diff < 5e-3f);
    }

    // 测试 BFloat16
    {
        std::vector<bfloat16_t> bf16_q(numel), bf16_k(numel), bf16_v(numel);
        convert_dtype(f32_q.data(), DataType::Float32, bf16_q.data(), DataType::BFloat16, numel);
        convert_dtype(f32_k.data(), DataType::Float32, bf16_k.data(), DataType::BFloat16, numel);
        convert_dtype(f32_v.data(), DataType::Float32, bf16_v.data(), DataType::BFloat16, numel);

        Graph g_bf16;
        auto q_bf = g_bf16.input({H, S, D}, DataType::BFloat16);
        auto k_bf = g_bf16.input({H, S, D}, DataType::BFloat16);
        auto v_bf = g_bf16.input({H, S, D}, DataType::BFloat16);
        auto out_bf = g_bf16.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q_bf, k_bf, v_bf);
        auto exec_bf = g_bf16.build(DeviceType::CPU);
        REQUIRE(exec_bf);

        q_bf.copy_from_host(as_bytes(bf16_q));
        k_bf.copy_from_host(as_bytes(bf16_k));
        v_bf.copy_from_host(as_bytes(bf16_v));
        exec_bf->execute();

        std::vector<bfloat16_t> res_bf(numel);
        out_bf.copy_to_host(as_writeable_bytes(res_bf));

        std::vector<float> res_f32(numel);
        convert_dtype(res_bf.data(), DataType::BFloat16, res_f32.data(), DataType::Float32, numel);
        require_all_finite(res_f32, "cpu_fused_attn_bf16");

        float max_diff = 0.0f;
        for (std::size_t i = 0; i < numel; ++i) {
            float diff = std::abs(golden[i] - res_f32[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(max_diff < 2e-2f);
    }
}

#if defined(VELOMIND_ENABLE_CUDA)
TEST_CASE("FusedAttention CUDA equivalence and precision", "[cuda][fused_attention]") {
    if (!Device::cuda().is_available()) {
        SUCCEED("CUDA device not available; skipping");
        return;
    }

    const dim_t H = 4;
    const dim_t S = 32;
    const dim_t D = 64;
    const std::size_t numel = static_cast<std::size_t>(H * S * D);

    auto f32_q = make_pattern_data(numel, 0.1f, 0.02f);
    auto f32_k = make_pattern_data(numel, -0.05f, 0.015f);
    auto f32_v = make_pattern_data(numel, 0.2f, 0.03f);

    // Float32 CUDA 对齐
    {
        Graph g_cpu;
        auto q_cpu = g_cpu.input({H, S, D}, DataType::Float32);
        auto k_cpu = g_cpu.input({H, S, D}, DataType::Float32);
        auto v_cpu = g_cpu.input({H, S, D}, DataType::Float32);
        auto out_cpu = g_cpu.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q_cpu, k_cpu, v_cpu);
        auto exec_cpu = g_cpu.build(DeviceType::CPU);
        REQUIRE(exec_cpu);

        Graph g_cuda;
        auto q_cuda = g_cuda.input({H, S, D}, DataType::Float32);
        auto k_cuda = g_cuda.input({H, S, D}, DataType::Float32);
        auto v_cuda = g_cuda.input({H, S, D}, DataType::Float32);
        auto out_cuda = g_cuda.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q_cuda, k_cuda, v_cuda);
        auto exec_cuda = g_cuda.build(DeviceType::CUDA);
        REQUIRE(exec_cuda);

        q_cpu.copy_from_host(as_bytes(f32_q));
        k_cpu.copy_from_host(as_bytes(f32_k));
        v_cpu.copy_from_host(as_bytes(f32_v));
        exec_cpu->execute();
        std::vector<float> cpu_res(numel);
        out_cpu.copy_to_host(as_writeable_bytes(cpu_res));

        q_cuda.copy_from_host(as_bytes(f32_q));
        k_cuda.copy_from_host(as_bytes(f32_k));
        v_cuda.copy_from_host(as_bytes(f32_v));
        exec_cuda->execute();
        std::vector<float> cuda_res(numel);
        out_cuda.copy_to_host(as_writeable_bytes(cuda_res));

        require_all_finite(cpu_res, "cpu_fused_attention_f32");
        require_all_finite(cuda_res, "cuda_fused_attention_f32");

        float max_diff = 0.0f;
        for (std::size_t i = 0; i < numel; ++i) {
            float diff = std::abs(cpu_res[i] - cuda_res[i]);
            if (diff > max_diff) max_diff = diff;
        }
        CHECK(max_diff < 1e-4f);
    }

    // Float16 CUDA 执行
    {
        std::vector<float16_t> f16_q(numel), f16_k(numel), f16_v(numel);
        convert_dtype(f32_q.data(), DataType::Float32, f16_q.data(), DataType::Float16, numel);
        convert_dtype(f32_k.data(), DataType::Float32, f16_k.data(), DataType::Float16, numel);
        convert_dtype(f32_v.data(), DataType::Float32, f16_v.data(), DataType::Float16, numel);

        Graph g;
        auto q = g.input({H, S, D}, DataType::Float16);
        auto k = g.input({H, S, D}, DataType::Float16);
        auto v = g.input({H, S, D}, DataType::Float16);
        auto out = g.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q, k, v);

        auto exec = g.build(DeviceType::CUDA);
        REQUIRE(exec);

        q.copy_from_host(as_bytes(f16_q));
        k.copy_from_host(as_bytes(f16_k));
        v.copy_from_host(as_bytes(f16_v));
        exec->execute();

        std::vector<float16_t> res_16(numel);
        out.copy_to_host(as_writeable_bytes(res_16));

        std::vector<float> res_f32(numel);
        convert_dtype(res_16.data(), DataType::Float16, res_f32.data(), DataType::Float32, numel);
        require_all_finite(res_f32, "cuda_fused_attention_f16");
    }

    // BFloat16 CUDA 执行
    {
        std::vector<bfloat16_t> bf16_q(numel), bf16_k(numel), bf16_v(numel);
        convert_dtype(f32_q.data(), DataType::Float32, bf16_q.data(), DataType::BFloat16, numel);
        convert_dtype(f32_k.data(), DataType::Float32, bf16_k.data(), DataType::BFloat16, numel);
        convert_dtype(f32_v.data(), DataType::Float32, bf16_v.data(), DataType::BFloat16, numel);

        Graph g;
        auto q = g.input({H, S, D}, DataType::BFloat16);
        auto k = g.input({H, S, D}, DataType::BFloat16);
        auto v = g.input({H, S, D}, DataType::BFloat16);
        auto out = g.op(
            OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
            q, k, v);

        auto exec = g.build(DeviceType::CUDA);
        REQUIRE(exec);

        q.copy_from_host(as_bytes(bf16_q));
        k.copy_from_host(as_bytes(bf16_k));
        v.copy_from_host(as_bytes(bf16_v));
        exec->execute();

        std::vector<bfloat16_t> res_bf(numel);
        out.copy_to_host(as_writeable_bytes(res_bf));

        std::vector<float> res_f32(numel);
        convert_dtype(res_bf.data(), DataType::BFloat16, res_f32.data(), DataType::Float32, numel);
        require_all_finite(res_f32, "cuda_fused_attention_bf16");
    }
}
#endif

TEST_CASE("FusedAttention memory and node reduction evaluation", "[graph][fused_attention][memory]") {
    const dim_t H = 4;
    const dim_t S = 256;
    const dim_t D = 64;

    // 未融合管道：Trans -> MatMul -> Add -> Softmax -> MatMul (5 个节点)
    Graph g_unfused;
    auto q_u = g_unfused.input({H, S, D}, DataType::Float32);
    auto k_u = g_unfused.input({H, S, D}, DataType::Float32);
    auto v_u = g_unfused.input({H, S, D}, DataType::Float32);
    auto mask_u = g_unfused.input({H, S, S}, DataType::Float32);

    auto kt_u = g_unfused.op(OpDescriptor{Op::Transpose, TransposeAttrs{{0, 2, 1}}}, k_u);
    auto scores_u = g_unfused.op(Op::MatMul, q_u, kt_u);
    auto masked_scores_u = g_unfused.op(Op::Add, scores_u, mask_u);
    auto weights_u = g_unfused.op(Op::Softmax, masked_scores_u);
    auto out_u = g_unfused.op(Op::MatMul, weights_u, v_u);
    g_unfused.mark_output(out_u);

    // 融合注意力管道：1 个 FusedAttention 节点
    Graph g_fused;
    auto q_f = g_fused.input({H, S, D}, DataType::Float32);
    auto k_f = g_fused.input({H, S, D}, DataType::Float32);
    auto v_f = g_fused.input({H, S, D}, DataType::Float32);
    auto out_f = g_fused.op(
        OpDescriptor{Op::FusedAttention, FusedAttentionAttrs{.scale = 1.0f, .is_causal = true}},
        q_f, k_f, v_f);
    g_fused.mark_output(out_f);

    // 验证节点数量缩减
    CHECK(g_fused.node_tot() == 1);
    CHECK(g_unfused.node_tot() == 5);

    auto exec_unfused = g_unfused.build(DeviceType::CPU);
    auto exec_fused = g_fused.build(DeviceType::CPU);
    REQUIRE(exec_unfused);
    REQUIRE(exec_fused);

    // 验证峰值内存占用大幅降低：未融合管道至少需要保留 [H, S, S] 尺寸的多份中间张量
    CHECK(exec_fused->memory_plan_stats().peak_bytes < exec_unfused->memory_plan_stats().peak_bytes);
}

TEST_CASE("FusedAttention LLaMA Graph prefill equivalence", "[llama][fused_attention]") {
    using namespace velomind::examples::llama;

    Config cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.max_seq_len = 16;
    cfg.rms_norm_eps = 1e-5f;

    const std::size_t seq = 4;
    const std::size_t H = cfg.num_heads;
    const std::size_t D = cfg.head_dim();

    std::vector<float> cos_buf, sin_buf;
    compute_rope_cache(seq, D, 10000.0f, cos_buf, sin_buf);
    std::vector<float> causal_mask_buf;
    compute_causal_mask(H, seq, causal_mask_buf);

    // 初始化权重张量生成器
    auto init_weights = [&](Graph& g, ForwardWeights& w, bool use_fused) {
        w.graph = &g;
        w.use_fused_attention = use_fused;
        w.w_embed = g.input({static_cast<dim_t>(cfg.vocab_size), static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32);
        w.w_lm_head = g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.vocab_size)}, DataType::Float32);
        w.w_final_norm = g.input({static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32);

        if (!use_fused) {
            w.causal_mask = g.input({static_cast<dim_t>(H), static_cast<dim_t>(seq), static_cast<dim_t>(seq)}, DataType::Float32);
        }
        w.cos_cache = g.input({static_cast<dim_t>(seq), static_cast<dim_t>(D / 2)}, DataType::Float32);
        w.sin_cache = g.input({static_cast<dim_t>(seq), static_cast<dim_t>(D / 2)}, DataType::Float32);

        for (std::size_t i = 0; i < cfg.num_layers; ++i) {
            w.w_attn_norm.push_back(g.input({static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32));
            w.wq.push_back(g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(H * D)}, DataType::Float32));
            w.wk.push_back(g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.num_kv_heads * D)}, DataType::Float32));
            w.wv.push_back(g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.num_kv_heads * D)}, DataType::Float32));
            w.wo.push_back(g.input({static_cast<dim_t>(H * D), static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32));

            w.w_ffn_norm.push_back(g.input({static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32));
            w.w_gate.push_back(g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.intermediate_size)}, DataType::Float32));
            w.w_up.push_back(g.input({static_cast<dim_t>(cfg.hidden_size), static_cast<dim_t>(cfg.intermediate_size)}, DataType::Float32));
            w.w_down.push_back(g.input({static_cast<dim_t>(cfg.intermediate_size), static_cast<dim_t>(cfg.hidden_size)}, DataType::Float32));
        }
    };

    Graph g_unfused;
    ForwardWeights w_unfused;
    init_weights(g_unfused, w_unfused, false);
    auto tokens_u = g_unfused.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto logits_u = build_forward_graph_prefill(g_unfused, cfg, tokens_u, w_unfused);

    Graph g_fused;
    ForwardWeights w_fused;
    init_weights(g_fused, w_fused, true);
    auto tokens_f = g_fused.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto logits_f = build_forward_graph_prefill(g_fused, cfg, tokens_f, w_fused);

    // 验证融入 FusedAttention 后的计算图节点明显少于未融合图
    CHECK(g_fused.node_tot() < g_unfused.node_tot());

    auto exec_unfused = g_unfused.build(DeviceType::CPU);
    auto exec_fused = g_fused.build(DeviceType::CPU);
    REQUIRE(exec_unfused);
    REQUIRE(exec_fused);

    // 填充相同输入与权重
    std::vector<int32_t> token_data = {1, 10, 5, 22};
    tokens_u.copy_from_host(as_bytes(token_data));
    tokens_f.copy_from_host(as_bytes(token_data));

    w_unfused.cos_cache.copy_from_host(as_bytes(cos_buf));
    w_unfused.sin_cache.copy_from_host(as_bytes(sin_buf));
    w_unfused.causal_mask.copy_from_host(as_bytes(causal_mask_buf));

    w_fused.cos_cache.copy_from_host(as_bytes(cos_buf));
    w_fused.sin_cache.copy_from_host(as_bytes(sin_buf));

    auto copy_param = [](Tensor& t1, Tensor& t2, float base) {
        auto d = make_pattern_data(storage_numel(*t1.storage()), base, 0.01f);
        t1.copy_from_host(as_bytes(d));
        t2.copy_from_host(as_bytes(d));
    };

    copy_param(w_unfused.w_embed, w_fused.w_embed, 0.1f);
    copy_param(w_unfused.w_lm_head, w_fused.w_lm_head, 0.05f);
    copy_param(w_unfused.w_final_norm, w_fused.w_final_norm, 1.0f);

    for (std::size_t i = 0; i < cfg.num_layers; ++i) {
        copy_param(w_unfused.w_attn_norm[i], w_fused.w_attn_norm[i], 1.0f);
        copy_param(w_unfused.wq[i], w_fused.wq[i], 0.02f);
        copy_param(w_unfused.wk[i], w_fused.wk[i], 0.02f);
        copy_param(w_unfused.wv[i], w_fused.wv[i], 0.02f);
        copy_param(w_unfused.wo[i], w_fused.wo[i], 0.02f);

        copy_param(w_unfused.w_ffn_norm[i], w_fused.w_ffn_norm[i], 1.0f);
        copy_param(w_unfused.w_gate[i], w_fused.w_gate[i], 0.02f);
        copy_param(w_unfused.w_up[i], w_fused.w_up[i], 0.02f);
        copy_param(w_unfused.w_down[i], w_fused.w_down[i], 0.02f);
    }

    exec_unfused->execute();
    exec_fused->execute();

    const std::size_t logits_numel = seq * cfg.vocab_size;
    std::vector<float> res_u(logits_numel);
    std::vector<float> res_f(logits_numel);
    logits_u.copy_to_host(as_writeable_bytes(res_u));
    logits_f.copy_to_host(as_writeable_bytes(res_f));

    require_all_finite(res_u, "llama_logits_unfused");
    require_all_finite(res_f, "llama_logits_fused");

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < logits_numel; ++i) {
        float diff = std::abs(res_u[i] - res_f[i]);
        if (diff > max_diff) max_diff = diff;
    }
    CHECK(max_diff < 1e-4f);
}
