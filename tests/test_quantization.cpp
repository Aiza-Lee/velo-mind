#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/quant.h"
#include "velomind/tensor.h"
#include "velomind/types.h"
#include "velomind/device.h"
#include "velomind/dtype.h"

#include "test_helpers.h"
#include "test_model_assets.h"
#include "llama_graph.h"
#include "llama_rope_mask.h"
#include "../examples/smollm2/model.h"
#include "../examples/smollm2/model_loader.h"
#include "../examples/tinyllama/model_loader.h"

#include "cpu/ops/quantized_matmul.h"

using namespace velomind;
using namespace velomind_test;
using namespace velomind::backend::cpu;

namespace {

inline auto generate_random_weights(std::size_t rows, std::size_t cols, float min_val = -1.0f, float max_val = 1.0f) -> std::vector<float> {
    std::vector<float> data(rows * cols);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& v : data) {
        v = dist(rng);
    }
    return data;
}

inline auto generate_activations(std::size_t m, std::size_t k) -> std::vector<float> {
    std::vector<float> data(m * k);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.05f * static_cast<float>((static_cast<int>(i % 23) - 11));
    }
    return data;
}

} // namespace

TEST_CASE("Op::QuantizedMatMul operator validation", "[validation][quantization]") {
    Graph g;
    auto a = g.input({2, 16, 64}, DataType::Float32);
    auto b_q = g.input({64, 128}, DataType::Int8);
    auto scales = g.input({128}, DataType::Float32);

    // 输入数量校验
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_q), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_q, scales, a), std::invalid_argument);

    // 秩与维度校验
    auto a_1d = g.input({64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a_1d, b_q, scales), std::invalid_argument);

    auto b_3d = g.input({2, 64, 128}, DataType::Int8);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_3d, scales), std::invalid_argument);

    // 内维 K 失配
    auto b_bad_k = g.input({32, 128}, DataType::Int8);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_bad_k, scales), std::invalid_argument);

    // 通道尺度 N 失配
    auto scales_bad_n = g.input({64}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_q, scales_bad_n), std::invalid_argument);

    // 分块尺度校验
    auto scales_block = g.input({2, 128}, DataType::Float32);
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::QuantizedMatMul, QuantizedMatMulAttrs{.quant_type = QuantType::Int8, .block_size = 32}},
                         a, b_q, scales_block));

    // 数据类型校验
    auto a_i32 = g.input({2, 16, 64}, DataType::Int32);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a_i32, b_q, scales), std::invalid_argument);

    auto b_f32 = g.input({64, 128}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::QuantizedMatMul, a, b_f32, scales), std::invalid_argument);

    // 正常构建签名
    REQUIRE_NOTHROW(g.op(Op::QuantizedMatMul, a, b_q, scales));
}

TEST_CASE("Quantization roundtrip and metrics (INT8 and INT4)", "[quantization][metrics]") {
    const std::size_t rows = 128;
    const std::size_t cols = 256;
    const auto orig = generate_random_weights(rows, cols, -1.5f, 1.5f);

    SECTION("INT8 per-channel symmetric quantization") {
        std::vector<std::int8_t> q_data(rows * cols);
        std::vector<float> scales(cols);
        std::vector<float> recon(rows * cols);

        quantize_weights_int8_per_channel(orig.data(), q_data.data(), scales.data(), rows, cols);
        dequantize_weights_int8_per_channel(q_data.data(), scales.data(), recon.data(), rows, cols);

        auto metrics = compute_quantization_metrics(orig.data(), recon.data(), rows * cols);
        REQUIRE(metrics.max_diff < 0.025f);
        REQUIRE(metrics.rmse < 0.01f);
        REQUIRE(metrics.cosine_sim > 0.999f);
        REQUIRE(metrics.snr_db > 40.0f);
    }

    SECTION("INT4 block symmetric quantization (block_size = 32)") {
        const std::size_t block_size = 32;
        const std::size_t num_blocks = (rows + block_size - 1) / block_size;
        const std::size_t packed_rows = (rows + 1) / 2;

        std::vector<std::uint8_t> packed_data(packed_rows * cols);
        std::vector<float> scales(num_blocks * cols);
        std::vector<float> recon(rows * cols);

        quantize_weights_int4_block(orig.data(), packed_data.data(), scales.data(), rows, cols, block_size);
        dequantize_weights_int4_block(packed_data.data(), scales.data(), recon.data(), rows, cols, block_size);

        auto metrics = compute_quantization_metrics(orig.data(), recon.data(), rows * cols);
        REQUIRE(metrics.max_diff < 0.15f);
        REQUIRE(metrics.rmse < 0.08f);
        REQUIRE(metrics.cosine_sim > 0.99f);
        REQUIRE(metrics.snr_db > 20.0f);
    }
}

TEST_CASE("QuantizedMatMul vs FP32 Reference MatMul numerical equivalence (CPU)", "[cpu][quantization]") {
    const std::vector<dim_t> test_m = {1, 4, 16};
    const std::vector<std::pair<dim_t, dim_t>> test_kn = {
        {64, 64},
        {128, 256},
        {576, 576},
        {576, 1536}
    };

    for (dim_t m : test_m) {
        for (const auto& [k, n] : test_kn) {
            DYNAMIC_SECTION("Shape M=" << m << " K=" << k << " N=" << n) {
                const auto w_orig = generate_random_weights(k, n, -1.0f, 1.0f);
                const auto a_data = generate_activations(m, k);

                std::vector<std::int8_t> q_data(k * n);
                std::vector<float> scales(n);
                std::vector<float> w_dequant(k * n);
                quantize_weights_int8_per_channel(w_orig.data(), q_data.data(), scales.data(), k, n);
                dequantize_weights_int8_per_channel(q_data.data(), scales.data(), w_dequant.data(), k, n);

                // 全精度 MatMul 基准
                Graph g_ref;
                auto a_ref = g_ref.input({m, k}, DataType::Float32);
                auto w_ref = g_ref.input({k, n}, DataType::Float32);
                auto c_ref = g_ref.op(Op::MatMul, a_ref, w_ref);
                auto exec_ref = g_ref.build(DeviceType::CPU);
                REQUIRE(exec_ref);

                a_ref.copy_from_host(as_bytes(a_data));
                w_ref.copy_from_host(as_bytes(w_dequant));
                exec_ref->execute();

                std::vector<float> out_ref(m * n);
                c_ref.copy_to_host(as_writeable_bytes(out_ref));
                require_all_finite(out_ref, "out_ref");

                // QuantizedMatMul 计算图
                Graph g_quant;
                auto a_q = g_quant.input({m, k}, DataType::Float32);
                auto w_q = g_quant.input({k, n}, DataType::Int8);
                auto s_q = g_quant.input({n}, DataType::Float32);
                auto c_q = g_quant.op(
                    OpDescriptor{Op::QuantizedMatMul, QuantizedMatMulAttrs{.quant_type = QuantType::Int8, .block_size = 0}},
                    a_q, w_q, s_q);
                auto exec_quant = g_quant.build(DeviceType::CPU);
                REQUIRE(exec_quant);

                a_q.copy_from_host(as_bytes(a_data));
                w_q.copy_from_host(as_bytes(q_data));
                s_q.copy_from_host(as_bytes(scales));
                exec_quant->execute();

                std::vector<float> out_quant(m * n);
                c_q.copy_to_host(as_writeable_bytes(out_quant));
                require_all_finite(out_quant, "out_quant");

                // 数值误差对比
                float max_diff = 0.0f;
                for (std::size_t i = 0; i < out_ref.size(); ++i) {
                    const float diff = std::abs(out_ref[i] - out_quant[i]);
                    if (diff > max_diff) max_diff = diff;
                }
                REQUIRE(max_diff < 1e-4f);
            }
        }
    }
}

TEST_CASE("QuantizedMatMul AVX2 optimized decode path vs scalar reference", "[cpu][quantization][avx2]") {
    const std::size_t k = 576;
    const std::size_t n = 1536;
    const auto w_orig = generate_random_weights(k, n, -1.0f, 1.0f);
    const auto a_data = generate_activations(1, k);

    std::vector<std::int8_t> q_data(k * n);
    std::vector<float> scales(n);
    quantize_weights_int8_per_channel(w_orig.data(), q_data.data(), scales.data(), k, n);

    std::vector<float> out_scalar(n, 0.0f);
    quantized_matmul_scalar_reference(
        a_data.data(), q_data.data(), scales.data(), out_scalar.data(),
        1, k, n,
        k, 1, n, 1, n, 1,
        QuantType::Int8, 0);
    require_all_finite(out_scalar, "out_scalar");

    std::vector<float> out_stream(n, 0.0f);
    bool ok = quantized_matmul_decode_f32(
        a_data.data(), q_data.data(), scales.data(), out_stream.data(),
        k, n, 1, n, 1,
        QuantType::Int8, 0);
    REQUIRE(ok);
    require_all_finite(out_stream, "out_stream");

    float max_diff = 0.0f;
    for (std::size_t j = 0; j < n; ++j) {
        float diff = std::abs(out_scalar[j] - out_stream[j]);
        if (diff > max_diff) max_diff = diff;
    }
    REQUIRE(max_diff < 1e-4f);
}

TEST_CASE("QuantizedMatMul Float16 and BFloat16 precision (CPU)", "[cpu][quantization][dtype]") {
    const std::size_t m = 2;
    const std::size_t k = 64;
    const std::size_t n = 64;

    const auto w_orig = generate_random_weights(k, n, -1.0f, 1.0f);
    const auto a_f32 = generate_activations(m, k);

    std::vector<std::int8_t> q_data(k * n);
    std::vector<float> scales_f32(n);
    quantize_weights_int8_per_channel(w_orig.data(), q_data.data(), scales_f32.data(), k, n);

    SECTION("Float16 activation and scale") {
        std::vector<float16_t> a_f16(m * k);
        for (std::size_t i = 0; i < a_f16.size(); ++i) a_f16[i] = float16_t(a_f32[i]);
        std::vector<float16_t> s_f16(n);
        for (std::size_t j = 0; j < n; ++j) s_f16[j] = float16_t(scales_f32[j]);

        Graph g;
        auto a = g.input({static_cast<dim_t>(m), static_cast<dim_t>(k)}, DataType::Float16);
        auto w = g.input({static_cast<dim_t>(k), static_cast<dim_t>(n)}, DataType::Int8);
        auto s = g.input({static_cast<dim_t>(n)}, DataType::Float16);
        auto c = g.op(Op::QuantizedMatMul, a, w, s);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec);

        a.copy_from_host(as_bytes(a_f16));
        w.copy_from_host(as_bytes(q_data));
        s.copy_from_host(as_bytes(s_f16));
        exec->execute();

        std::vector<float16_t> out_f16(m * n);
        c.copy_to_host(as_writeable_bytes(out_f16));
        for (auto v : out_f16) {
            REQUIRE(std::isfinite(static_cast<float>(v)));
        }
    }

    SECTION("BFloat16 activation and scale") {
        std::vector<bfloat16_t> a_bf16(m * k);
        for (std::size_t i = 0; i < a_bf16.size(); ++i) a_bf16[i] = bfloat16_t(a_f32[i]);
        std::vector<bfloat16_t> s_bf16(n);
        for (std::size_t j = 0; j < n; ++j) s_bf16[j] = bfloat16_t(scales_f32[j]);

        Graph g;
        auto a = g.input({static_cast<dim_t>(m), static_cast<dim_t>(k)}, DataType::BFloat16);
        auto w = g.input({static_cast<dim_t>(k), static_cast<dim_t>(n)}, DataType::Int8);
        auto s = g.input({static_cast<dim_t>(n)}, DataType::BFloat16);
        auto c = g.op(Op::QuantizedMatMul, a, w, s);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec);

        a.copy_from_host(as_bytes(a_bf16));
        w.copy_from_host(as_bytes(q_data));
        s.copy_from_host(as_bytes(s_bf16));
        exec->execute();

        std::vector<bfloat16_t> out_bf16(m * n);
        c.copy_to_host(as_writeable_bytes(out_bf16));
        for (auto v : out_bf16) {
            REQUIRE(std::isfinite(static_cast<float>(v)));
        }
    }
}

#if VELOMIND_ENABLE_CUDA
TEST_CASE("QuantizedMatMul CUDA hardware acceleration and cross-backend alignment", "[cuda][quantization]") {
    if (!Device::cuda().is_available()) {
        SKIP("CUDA device not available");
    }

    const std::vector<dim_t> test_m = {1, 4, 16};
    const dim_t k = 576;
    const dim_t n = 576;

    const auto w_orig = generate_random_weights(k, n, -1.0f, 1.0f);
    std::vector<std::int8_t> q_data(k * n);
    std::vector<float> scales(n);
    quantize_weights_int8_per_channel(w_orig.data(), q_data.data(), scales.data(), k, n);

    for (dim_t m : test_m) {
        DYNAMIC_SECTION("CUDA vs CPU M=" << m) {
            const auto a_data = generate_activations(m, k);

            // CPU 基准
            Graph g_cpu;
            auto a_c = g_cpu.input({m, k}, DataType::Float32);
            auto w_c = g_cpu.input({k, n}, DataType::Int8);
            auto s_c = g_cpu.input({n}, DataType::Float32);
            auto out_c = g_cpu.op(Op::QuantizedMatMul, a_c, w_c, s_c);
            auto exec_cpu = g_cpu.build(DeviceType::CPU);
            REQUIRE(exec_cpu);

            a_c.copy_from_host(as_bytes(a_data));
            w_c.copy_from_host(as_bytes(q_data));
            s_c.copy_from_host(as_bytes(scales));
            exec_cpu->execute();

            std::vector<float> res_cpu(m * n);
            out_c.copy_to_host(as_writeable_bytes(res_cpu));
            require_all_finite(res_cpu, "res_cpu");

            // CUDA 执行
            Graph g_cuda;
            auto a_d = g_cuda.input({m, k}, DataType::Float32);
            auto w_d = g_cuda.input({k, n}, DataType::Int8);
            auto s_d = g_cuda.input({n}, DataType::Float32);
            auto out_d = g_cuda.op(Op::QuantizedMatMul, a_d, w_d, s_d);
            auto exec_cuda = g_cuda.build(DeviceType::CUDA);
            REQUIRE(exec_cuda);

            a_d.copy_from_host(as_bytes(a_data));
            w_d.copy_from_host(as_bytes(q_data));
            s_d.copy_from_host(as_bytes(scales));
            exec_cuda->execute();

            std::vector<float> res_cuda(m * n);
            out_d.copy_to_host(as_writeable_bytes(res_cuda));
            require_all_finite(res_cuda, "res_cuda");

            float max_diff = 0.0f;
            for (std::size_t i = 0; i < res_cpu.size(); ++i) {
                float diff = std::abs(res_cpu[i] - res_cuda[i]);
                if (diff > max_diff) max_diff = diff;
            }
            REQUIRE(max_diff < 1e-4f);
        }
    }
}
#endif

TEST_CASE("Quantization Memory Footprint Verification", "[memory][quantization]") {
    const std::size_t rows = 576;
    const std::size_t cols = 1536;
    const std::size_t numel = rows * cols;

    // 单张量物理权重字节对比
    const std::size_t fp32_bytes = numel * sizeof(float);
    const std::size_t int8_bytes = numel * sizeof(std::int8_t);
    const std::size_t int4_bytes = ((rows + 1) / 2) * cols;

    REQUIRE(fp32_bytes == numel * 4);
    REQUIRE(int8_bytes == numel * 1);
    REQUIRE(fp32_bytes / int8_bytes == 4);
    REQUIRE(fp32_bytes / int4_bytes == 8);

    // 权重共享与内存规划验证
    Graph g_fp;
    auto in_fp = g_fp.input({1, static_cast<dim_t>(rows)}, DataType::Float32);
    auto w_fp  = g_fp.input({static_cast<dim_t>(rows), static_cast<dim_t>(cols)}, DataType::Float32);
    auto out_fp = g_fp.op(Op::MatMul, in_fp, w_fp);
    auto exec_fp = g_fp.build(DeviceType::CPU);
    REQUIRE(exec_fp);

    Graph g_q;
    auto in_q = g_q.input({1, static_cast<dim_t>(rows)}, DataType::Float32);
    auto w_q  = g_q.input({static_cast<dim_t>(rows), static_cast<dim_t>(cols)}, DataType::Int8);
    auto s_q  = g_q.input({static_cast<dim_t>(cols)}, DataType::Float32);
    auto out_q = g_q.op(Op::QuantizedMatMul, in_q, w_q, s_q);
    auto exec_q = g_q.build(DeviceType::CPU);
    REQUIRE(exec_q);

    const auto* w_fp_stor = w_fp.shared_storage().get();
    const auto* w_q_stor  = w_q.shared_storage().get();
    REQUIRE(w_fp_stor->size_bytes == rows * cols * 4);
    REQUIRE(w_q_stor->size_bytes == rows * cols * 1);
}

TEST_CASE("Quantization Single-Token Decode Throughput Verification", "[benchmark][quantization]") {
    const std::size_t k = 576;
    const std::size_t n = 1536;

    const auto w_orig = generate_random_weights(k, n, -1.0f, 1.0f);
    const auto a_data = generate_activations(1, k);

    std::vector<std::int8_t> q_data(k * n);
    std::vector<float> scales(n);
    quantize_weights_int8_per_channel(w_orig.data(), q_data.data(), scales.data(), k, n);

    Graph g_quant;
    auto a_q = g_quant.input({1, static_cast<dim_t>(k)}, DataType::Float32);
    auto w_q = g_quant.input({static_cast<dim_t>(k), static_cast<dim_t>(n)}, DataType::Int8);
    auto s_q = g_quant.input({static_cast<dim_t>(n)}, DataType::Float32);
    auto c_q = g_quant.op(Op::QuantizedMatMul, a_q, w_q, s_q);
    auto exec = g_quant.build(DeviceType::CPU);
    REQUIRE(exec);

    a_q.copy_from_host(as_bytes(a_data));
    w_q.copy_from_host(as_bytes(q_data));
    s_q.copy_from_host(as_bytes(scales));

    // 预热
    for (int i = 0; i < 10; ++i) {
        exec->execute();
    }

    // 吞吐基准测试
    const int iterations = 100;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        exec->execute();
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

    REQUIRE(elapsed_us > 0.0);
    const double tokens_per_sec = 1e6 / elapsed_us;
    REQUIRE(tokens_per_sec > 100.0);
}

TEST_CASE("Quantization End-to-End Model Quality and Logits Verification (toy_llama)", "[e2e][quantization]") {
    const auto model_path = velomind_test::resolve_repo_path("tests/testdata/toy_llama/model.safetensors");
    velomind_test::require_or_skip_asset(model_path, "toy_llama model safetensors");

    auto sf_opt = load_safetensors(model_path);
    REQUIRE(sf_opt.has_value());
    const auto& sf = *sf_opt;

    // Toy LLaMA 配置
    velomind::examples::llama::Config cfg;
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.rms_norm_eps = 1e-5f;
    cfg.rope_theta = 10000.0f;
    cfg.tie_word_embeddings = true;

    const std::size_t seq = 4;
    std::vector<std::int32_t> prompt_tokens = {1, 10, 20, 30};

    // FP32 浮点推理参考
    Graph g_fp;
    auto tokens_fp = g_fp.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto w_fp = velomind::examples::tinyllama::bind_tinyllama_weights(g_fp, cfg, seq);
    auto logits_fp = build_forward_graph_prefill(g_fp, cfg, tokens_fp, w_fp, nullptr, true);
    auto exec_fp = g_fp.build(DeviceType::CPU);
    REQUIRE(exec_fp);

    tokens_fp.copy_from_host(as_bytes(prompt_tokens));
    velomind::examples::tinyllama::populate_tinyllama_weights(sf, cfg, w_fp, seq);
    exec_fp->execute();

    std::vector<float> ref_logits(cfg.vocab_size);
    logits_fp.copy_to_host(as_writeable_bytes(ref_logits));
    require_all_finite(ref_logits, "ref_logits");

    // INT8 模型量化推理
    Graph g_q;
    auto tokens_q = g_q.input({static_cast<dim_t>(seq)}, DataType::Int32);
    auto w_quant = quantize_forward_weights(g_q, w_fp, QuantType::Int8, 0, DeviceType::CPU);
    auto logits_q = build_forward_graph_prefill(g_q, cfg, tokens_q, w_quant, nullptr, true);
    auto exec_q = g_q.build(DeviceType::CPU);
    REQUIRE(exec_q);

    tokens_q.copy_from_host(as_bytes(prompt_tokens));
    exec_q->execute();

    std::vector<float> q_logits(cfg.vocab_size);
    logits_q.copy_to_host(as_writeable_bytes(q_logits));
    require_all_finite(q_logits, "q_logits");

    // 验收模型保真度与预测一致性
    auto metrics = compute_quantization_metrics(ref_logits.data(), q_logits.data(), cfg.vocab_size);
    REQUIRE(metrics.cosine_sim > 0.999f);
    REQUIRE(metrics.rmse < 0.15f);

    auto argmax_fn = [](const std::vector<float>& l) -> std::int32_t {
        return static_cast<std::int32_t>(std::distance(l.begin(), std::max_element(l.begin(), l.end())));
    };

    std::int32_t pred_fp = argmax_fn(ref_logits);
    std::int32_t pred_q  = argmax_fn(q_logits);
    REQUIRE(pred_fp == pred_q);
}
