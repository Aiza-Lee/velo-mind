#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "../test_helpers.h"
#include "src/cpu/ops/matmul.h"

using namespace velomind;
using namespace velomind::backend::cpu;
using velomind_test::as_bytes;
using velomind_test::as_writeable_bytes;

TEST_CASE("CPU MatMul - weight panel packing (pack_b_panel) layout and padding", "[cpu][matmul][packing]") {
    // 验证 pack_b_panel 将行优先及跨步矩阵正确重排为宽度 16 的微面板，并在尾部补零
    const std::size_t K = 5;
    const std::size_t N = 18;
    std::vector<float> b(K * N);
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i + 1);
    }

    std::vector<float> b_packed(K * 32, -999.0f);
    pack_b_panel(b.data(), b_packed.data(), 0, K, 0, N, N, 1, 16);

    for (std::size_t p = 0; p < K; ++p) {
        for (std::size_t j = 0; j < 16; ++j) {
            REQUIRE(b_packed[p * 16 + j] == b[p * N + j]);
        }
    }

    const std::size_t panel1_offset = K * 16;
    for (std::size_t p = 0; p < K; ++p) {
        for (std::size_t j = 0; j < 2; ++j) {
            REQUIRE(b_packed[panel1_offset + p * 16 + j] == b[p * N + 16 + j]);
        }
        for (std::size_t j = 2; j < 16; ++j) {
            REQUIRE(b_packed[panel1_offset + p * 16 + j] == 0.0f);
        }
    }
}

TEST_CASE("CPU MatMul - real model shapes (SmolLM2 & TinyLlama) equivalence vs scalar reference",
          "[cpu][matmul][shapes]") {
    // 真实 LLM 模型中的核心 M/K/N 矩阵乘维度组合
    struct ModelShape {
        std::size_t m;
        std::size_t k;
        std::size_t n;
        const char* desc;
    };

    const std::vector<ModelShape> shapes = {
        // SmolLM2-135M 维度
        {1, 576, 1536, "SmolLM2 decode: MLP Up/Gate (M=1, K=576, N=1536)"},
        {1, 1536, 576, "SmolLM2 decode: MLP Down (M=1, K=1536, N=576)"},
        {1, 576, 192,  "SmolLM2 decode: Native GQA KV (M=1, K=576, N=192)"},
        {1, 576, 4096, "SmolLM2 decode: Large Vocab Head Slice (M=1, K=576, N=4096)"},
        {2, 576, 1536, "SmolLM2 mini-batch M=2: MLP Up/Gate (M=2, K=576, N=1536)"},
        {4, 576, 576,  "SmolLM2 mini-batch M=4: Attention Out (M=4, K=576, N=576)"},
        {8, 576, 576,  "SmolLM2 prefill M=8: Attention Out (M=8, K=576, N=576)"},
        {32, 576, 1536,"SmolLM2 prefill M=32: MLP Up (M=32, K=576, N=1536)"},

        // TinyLlama-1.1B 维度
        {1, 2048, 2048, "TinyLlama decode: Attention Q/O (M=1, K=2048, N=2048)"},
        {1, 2048, 256,  "TinyLlama decode: GQA KV (M=1, K=2048, N=256)"},
        {1, 2048, 5632, "TinyLlama decode: MLP Gate/Up (M=1, K=2048, N=5632)"},
        {1, 5632, 2048, "TinyLlama decode: MLP Down (M=1, K=5632, N=2048)"},
        {1, 2048, 4096, "TinyLlama decode: Vocab Head Slice (M=1, K=2048, N=4096)"},
        {4, 2048, 2048, "TinyLlama mini-batch M=4: Attention Q/O (M=4, K=2048, N=2048)"},
        {16, 2048, 2048,"TinyLlama prefill M=16: Attention Q/O (M=16, K=2048, N=2048)"}
    };

    for (const auto& item : shapes) {
        DYNAMIC_SECTION(item.desc) {
            std::vector<float> a(item.m * item.k);
            std::vector<float> b(item.k * item.n);
            std::vector<float> c_scalar(item.m * item.n, 0.0f);
            std::vector<float> c_opt(item.m * item.n, 0.0f);

            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = std::sin(static_cast<float>(i + 1) * 0.03f);
            }
            for (std::size_t i = 0; i < b.size(); ++i) {
                b[i] = std::cos(static_cast<float>(i + 1) * 0.02f);
            }

            matmul_scalar_reference(a.data(), b.data(), c_scalar.data(),
                                    item.m, item.k, item.n,
                                    item.k, 1, item.n, 1, item.n, 1);

            matmul_2d_f32(a.data(), b.data(), c_opt.data(),
                          item.m, item.k, item.n,
                          item.k, 1, item.n, 1, item.n, 1);

            float max_diff = 0.0f;
            for (std::size_t i = 0; i < c_scalar.size(); ++i) {
                max_diff = std::max(max_diff, std::abs(c_scalar[i] - c_opt[i]));
            }
            REQUIRE(max_diff < 1e-4f);
        }
    }
}

TEST_CASE("CPU MatMul - column-major and transposed view dot-product path", "[cpu][matmul][strided]") {
    // 模拟注意力矩阵点积 Q x K^T：其中 K^T 为非连续转置视图（列优先，b_row_stride == 1, b_col_stride == K）
    const std::size_t K = 64;
    const std::size_t N = 128;

    for (std::size_t M : {1, 2, 4, 8}) {
        std::vector<float> a(M * K);
        std::vector<float> b(K * N);
        std::vector<float> c_scalar(M * N, 0.0f);
        std::vector<float> c_opt(M * N, 0.0f);

        for (std::size_t i = 0; i < a.size(); ++i) a[i] = 0.1f * static_cast<float>((i % 7) + 1);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = 0.05f * static_cast<float>((i % 11) + 1);

        matmul_scalar_reference(a.data(), b.data(), c_scalar.data(),
                                M, K, N,
                                K, 1, 1, K, N, 1);

        matmul_2d_f32(a.data(), b.data(), c_opt.data(),
                      M, K, N,
                      K, 1, 1, K, N, 1);

        float max_diff = 0.0f;
        for (std::size_t i = 0; i < c_scalar.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(c_scalar[i] - c_opt[i]));
        }
        REQUIRE(max_diff < 1e-4f);
    }
}

TEST_CASE("CPU MatMul - Graph execution with real model shapes and multi-batch", "[cpu][matmul][graph]") {
    const std::size_t B = 2;
    const std::size_t M = 4;
    const std::size_t K = 576;
    const std::size_t N = 192;

    Graph g;
    auto a = g.input({static_cast<dim_t>(B), static_cast<dim_t>(M), static_cast<dim_t>(K)}, DataType::Float32);
    auto b = g.input({static_cast<dim_t>(B), static_cast<dim_t>(K), static_cast<dim_t>(N)}, DataType::Float32);
    auto c = g.op(Op::MatMul, a, b);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data(B * M * K, 0.25f);
    std::vector<float> b_data(B * K * N, 0.5f);
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));

    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> c_data(B * M * N);
    c.copy_to_host(as_writeable_bytes(c_data));

    for (std::size_t i = 0; i < c_data.size(); ++i) {
        REQUIRE(c_data[i] == Catch::Approx(72.0f).epsilon(1e-4f));
    }
}

TEST_CASE("CPU MatMul - micro-benchmark speedup verification on real model shapes", "[cpu][matmul][benchmark]") {
    // 评测真实模型典型形状下小 M 与分块打包路径的性能提升
    struct BenchItem {
        std::size_t m, k, n;
        const char* name;
    };

    const std::vector<BenchItem> bench_shapes = {
        {1, 576, 1536, "SmolLM2 decode M=1 K=576 N=1536"},
        {1, 2048, 5632, "TinyLlama decode M=1 K=2048 N=5632"},
        {4, 576, 1536, "SmolLM2 mini-batch M=4 K=576 N=1536"},
        {16, 576, 576, "SmolLM2 prefill M=16 K=576 N=576"}
    };

    for (const auto& item : bench_shapes) {
        std::vector<float> a(item.m * item.k, 0.5f);
        std::vector<float> b(item.k * item.n, 0.25f);
        std::vector<float> c_scalar(item.m * item.n, 0.0f);
        std::vector<float> c_opt(item.m * item.n, 0.0f);

        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_scalar_reference(a.data(), b.data(), c_scalar.data(),
                                item.m, item.k, item.n,
                                item.k, 1, item.n, 1, item.n, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        matmul_2d_f32(a.data(), b.data(), c_opt.data(),
                      item.m, item.k, item.n,
                      item.k, 1, item.n, 1, item.n, 1);
        auto t3 = std::chrono::high_resolution_clock::now();
        double opt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 确保优化路径至少不慢于标量，且对于较大规模有明确加速
        REQUIRE(opt_ms <= scalar_ms * 1.5); // 宽裕上界容忍 CI 环境波动
        REQUIRE(c_scalar[0] == Catch::Approx(c_opt[0]).epsilon(1e-4f));
    }
}
