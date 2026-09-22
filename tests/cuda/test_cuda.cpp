#include <vector>
#include <cstdlib>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "../test_helpers.h"

#ifdef VELOMIND_ENABLE_CUDA

#include "cuda/cuda_context.h"

namespace {
struct ScopedBackendFailure {
    std::string previous;
    bool had_previous;

    explicit ScopedBackendFailure(const char* spec) {
        const char* value = std::getenv("VELOMIND_FAIL_BACKEND");
        had_previous = (value != nullptr);
        if (had_previous) previous = value;
        setenv("VELOMIND_FAIL_BACKEND", spec, 1);
    }

    ~ScopedBackendFailure() {
        if (had_previous) setenv("VELOMIND_FAIL_BACKEND", previous.c_str(), 1);
        else unsetenv("VELOMIND_FAIL_BACKEND");
    }
};

inline auto build_cuda(velomind::Graph& g) -> std::unique_ptr<velomind::Executable> {
    if (!velomind::Device::cuda().is_available()) {
        return nullptr;
    }
    try {
        return g.build(velomind::DeviceType::CUDA);
    } catch (const std::exception&) {
        return nullptr;
    }
}
}

TEST_CASE("CUDA errors identify operator stage and output shape", "[cuda][failure]") {
    using namespace velomind;
    using velomind_test::as_bytes;

    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");
    Graph g;
    auto a = g.input({2, 2}, DataType::Float32);
    auto b = g.input({2, 2}, DataType::Float32);
    g.op(Op::Add, a, b);
    auto exec = g.build(DeviceType::CUDA);
    a.copy_from_host(as_bytes(std::vector<float>{1, 2, 3, 4}));
    b.copy_from_host(as_bytes(std::vector<float>{5, 6, 7, 8}));
    {
        ScopedBackendFailure failure("CUDA:Add:sync");
        try {
            exec->execute();
            FAIL("应报告注入的同步失败");
        } catch (const std::runtime_error& error) {
            REQUIRE(std::string(error.what()).find("Add on CUDA output shape=[2,2] at sync") != std::string::npos);
        }
    }
    REQUIRE_NOTHROW(exec->execute());

    Graph matmul;
    auto left = matmul.input({2, 2}, DataType::Float32);
    auto right = matmul.input({2, 2}, DataType::Float32);
    matmul.op(Op::MatMul, left, right);
    auto matmul_exec = matmul.build(DeviceType::CUDA);
    left.copy_from_host(as_bytes(std::vector<float>{1, 2, 3, 4}));
    right.copy_from_host(as_bytes(std::vector<float>{5, 6, 7, 8}));
    {
        ScopedBackendFailure failure("CUDA:MatMul:gemm");
        try {
            matmul_exec->execute();
            FAIL("应报告注入的 GEMM 失败");
        } catch (const std::runtime_error& error) {
            REQUIRE(std::string(error.what()).find("MatMul on CUDA output shape=[2,2] at gemm") != std::string::npos);
        }
    }
    REQUIRE_NOTHROW(matmul_exec->execute());
}

TEST_CASE("Element-wise Add round-trip (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Add, a, b);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {10, 20, 30, 40};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));
    REQUIRE(c_data[0] == 11.0f);
    REQUIRE(c_data[1] == 22.0f);
    REQUIRE(c_data[2] == 33.0f);
    REQUIRE(c_data[3] == 44.0f);
}

TEST_CASE("Element-wise Sub round-trip (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Sub, a, b);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> a_data = {10, 20, 30, 40};
    std::vector<float> b_data = {1, 2, 3, 4};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));
    REQUIRE(c_data[0] == 9.0f);
    REQUIRE(c_data[1] == 18.0f);
    REQUIRE(c_data[2] == 27.0f);
    REQUIRE(c_data[3] == 36.0f);
}

TEST_CASE("Element-wise Mul round-trip (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Mul, a, b);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {10, 20, 30, 40};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));
    REQUIRE(c_data[0] == 10.0f);
    REQUIRE(c_data[1] == 40.0f);
    REQUIRE(c_data[2] == 90.0f);
    REQUIRE(c_data[3] == 160.0f);
}

TEST_CASE("Relu activation (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4}, DataType::Float32);
    auto y = g.op(Op::Relu, x);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {-1, 0, 1, 2};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == 0.0f);
    REQUIRE(y_data[1] == 0.0f);
    REQUIRE(y_data[2] == 1.0f);
    REQUIRE(y_data[3] == 2.0f);
}

TEST_CASE("MatMul round-trip (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto W = g.input({3, 2}, DataType::Float32);
    auto y = g.op(Op::MatMul, x, W);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3,
                                  4, 5, 6};
    std::vector<float> W_data = {1, 0,
                                  0, 1,
                                  1, 1};
    x.copy_from_host(as_bytes(x_data));
    W.copy_from_host(as_bytes(W_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == 4.0f);
    REQUIRE(y_data[1] == 5.0f);
    REQUIRE(y_data[2] == 10.0f);
    REQUIRE(y_data[3] == 11.0f);
}

TEST_CASE("Batched MatMul round-trip (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 2, 3}, DataType::Float32);
    auto W = g.input({2, 3, 2}, DataType::Float32);
    auto y = g.op(Op::MatMul, x, W);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {
        1, 2, 3,  4, 5, 6,
        1, 0, 1,  0, 2, 0
    };
    std::vector<float> W_data = {
        1, 0,  0, 1,  1, 1,
        2, 1,  0, 2,  1, 0
    };
    x.copy_from_host(as_bytes(x_data));
    W.copy_from_host(as_bytes(W_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data.size() == 8);
    REQUIRE(y_data[0] == 4.0f);
    REQUIRE(y_data[1] == 5.0f);
    REQUIRE(y_data[2] == 10.0f);
    REQUIRE(y_data[3] == 11.0f);
}

TEST_CASE("Embedding lookup (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto table = g.input({4, 3}, DataType::Float32);
    auto indices = g.input({2}, DataType::Int32);
    auto y = g.op(Op::Embedding, table, indices);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> table_data = {
        0.1f, 0.2f, 0.3f,
        1.1f, 1.2f, 1.3f,
        2.1f, 2.2f, 2.3f,
        3.1f, 3.2f, 3.3f,
    };
    std::vector<std::int32_t> idx_data = {2, 0};
    table.copy_from_host(as_bytes(table_data));
    indices.copy_from_host(as_bytes(idx_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data.size() == 6);
    REQUIRE(y_data[0] == 2.1f);
    REQUIRE(y_data[1] == 2.2f);
    REQUIRE(y_data[2] == 2.3f);
    REQUIRE(y_data[3] == 0.1f);
    REQUIRE(y_data[4] == 0.2f);
    REQUIRE(y_data[5] == 0.3f);
}

TEST_CASE("RMSNorm (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 4}, DataType::Float32);
    auto w = g.input({4}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{1e-5f, -1}}, x, w);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {1.0f, 1.0f, 1.0f, 1.0f,
                                 2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> w_data = {1.0f, 2.0f, 1.0f, 2.0f};
    x.copy_from_host(as_bytes(x_data));
    w.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data.size() == 8);
    REQUIRE(std::abs(y_data[0] - 1.0f) < 1e-4f);
    REQUIRE(std::abs(y_data[1] - 2.0f) < 1e-4f);
    REQUIRE(std::abs(y_data[4] - 1.0f) < 1e-4f);
    REQUIRE(std::abs(y_data[5] - 2.0f) < 1e-4f);
}

TEST_CASE("RotaryEmbedding (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x   = g.input({1, 2, 4}, DataType::Float32);
    auto cos = g.input({2, 2},    DataType::Float32);
    auto sin = g.input({2, 2},    DataType::Float32);
    auto y   = g.op(Op::RotaryEmbedding, x, cos, sin);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 1.0f, 4.0f, 3.0f
    };
    std::vector<float> cos_data = {1.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> sin_data = {0.0f, 0.0f, 1.0f, 1.0f};

    x.copy_from_host(as_bytes(x_data));
    cos.copy_from_host(as_bytes(cos_data));
    sin.copy_from_host(as_bytes(sin_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data.size() == 8);

    REQUIRE(std::abs(y_data[0] - 1.0f) < 1e-5f);
    REQUIRE(std::abs(y_data[1] - 2.0f) < 1e-5f);
    REQUIRE(std::abs(y_data[2] - 3.0f) < 1e-5f);
    REQUIRE(std::abs(y_data[3] - 4.0f) < 1e-5f);

    REQUIRE(std::abs(y_data[4] - (-4.0f)) < 1e-5f);
    REQUIRE(std::abs(y_data[5] - (-3.0f)) < 1e-5f);
    REQUIRE(std::abs(y_data[6] - 2.0f) < 1e-5f);
    REQUIRE(std::abs(y_data[7] - 1.0f) < 1e-5f);
}

TEST_CASE("Reshape & Transpose (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({2, 3}, DataType::Float32);
    auto b = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{6}}}, a);
    auto c = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{3, 2}}}, b);
    auto d = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0}}}, c);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    a.copy_from_host(as_bytes(a_data));
    exec->execute();

    std::vector<float> d_data(d.numel());
    d.copy_to_host(as_writeable_bytes(d_data));
    REQUIRE(d_data.size() == 6);

    REQUIRE(d_data[0] == 1.0f);
    REQUIRE(d_data[1] == 3.0f);
    REQUIRE(d_data[2] == 5.0f);
    REQUIRE(d_data[3] == 2.0f);
    REQUIRE(d_data[4] == 4.0f);
    REQUIRE(d_data[5] == 6.0f);
}

TEST_CASE("Rsqrt (Float32, CUDA)", "[graph][cuda]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({3}, DataType::Float32);
    auto y = g.op(Op::Rsqrt, x);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {1.0f, 4.0f, 16.0f};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(std::abs(y_data[0] - 1.0f) < 1e-5f);
    REQUIRE(std::abs(y_data[1] - 0.5f) < 1e-5f);
    REQUIRE(std::abs(y_data[2] - 0.25f) < 1e-5f);
}

TEST_CASE("RepeatKV (Float32, CUDA)", "[graph][cuda][repeat_kv]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 2, 2}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{3, 0}}, x);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y.shape() == shape_t{6, 2, 2});
    std::vector<float> expected = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };
    REQUIRE(y_data == expected);
}

TEST_CASE("Slice (Float32, CUDA) - axis 0 last row and partial D2H", "[cuda][slice][partial_copy]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4, 3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {3, 0}, .ends = {4, 3}, .strides = {1, 1}
    }}, x);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    // 验证 CUDA Slice 算子输出为第 3 行
    REQUIRE(y.shape() == shape_t{1, 3});
    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    std::vector<float> expected = {10.f, 11.f, 12.f};
    REQUIRE(y_data == expected);

    // 验证直接对 x 执行局部 D2H 读取末行，结果与 Slice 完全一致
    std::vector<float> direct_last_row(3);
    x.copy_to_host(std::as_writable_bytes(std::span(direct_last_row)), 9 * sizeof(float));
    REQUIRE(direct_last_row == expected);
}

TEST_CASE("Slice (Float32, CUDA) - multi-axis striding and negative indices", "[cuda][slice]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({5, 6}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {-4, 1}, .ends = {-1, 5}, .strides = {2, 2}
    }}, x);

    auto exec = build_cuda(g);
    if (exec == nullptr) SKIP("CUDA backend not available in this build.");

    const std::vector<float> x_data = [] {
        std::vector<float> v(30);
        for (std::size_t i = 0; i < 30; ++i) v[i] = static_cast<float>(i);
        return v;
    }();
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    REQUIRE(y.shape() == shape_t{2, 2});
    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    std::vector<float> expected = {7.f, 9.f, 19.f, 21.f};
    REQUIRE(y_data == expected);
}

TEST_CASE("CUDA Context - RAII stream, cuBLAS handle, and scoping", "[cuda][context]") {
    using namespace velomind;
    using namespace velomind::backend::cuda;
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");

    // 验证 CudaContext 资源分配、流有效性与显式同步
    {
        CudaContext ctx;
        REQUIRE(ctx.stream_handle() != nullptr);
        REQUIRE(ctx.cublas() != nullptr);
        REQUIRE_NOTHROW(ctx.synchronize());
    }

    // 验证 ScopedCudaContext 覆盖与自动还原生命周期
    {
        CudaContext custom_ctx;
        auto* custom_stream = custom_ctx.stream_handle();
        auto* default_stream = get_cuda_context().stream_handle();
        REQUIRE(custom_stream != default_stream);

        {
            ScopedCudaContext scope(&custom_ctx);
            REQUIRE(get_cuda_context().stream_handle() == custom_stream);
            REQUIRE(get_cuda_context().cublas() == custom_ctx.cublas());
        }

        REQUIRE(get_cuda_context().stream_handle() == default_stream);
    }
}

TEST_CASE("CUDA MatMul - 4D Strided Batched GEMM (Attention pattern)", "[cuda][matmul][batched]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");

    // [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S] 注意力评分矩阵乘法
    const dim_t B = 2, H = 3, S = 4, D = 8;
    Graph g_cpu;
    auto q_cpu = g_cpu.input({B, H, S, D}, DataType::Float32);
    auto k_cpu = g_cpu.input({B, H, D, S}, DataType::Float32);
    auto out_cpu = g_cpu.op(Op::MatMul, q_cpu, k_cpu);
    auto exec_cpu = g_cpu.build(DeviceType::CPU);

    Graph g_cuda;
    auto q_cuda = g_cuda.input({B, H, S, D}, DataType::Float32);
    auto k_cuda = g_cuda.input({B, H, D, S}, DataType::Float32);
    auto out_cuda = g_cuda.op(Op::MatMul, q_cuda, k_cuda);
    auto exec_cuda = g_cuda.build(DeviceType::CUDA);

    const std::size_t num_q = static_cast<std::size_t>(B * H * S * D);
    const std::size_t num_k = static_cast<std::size_t>(B * H * D * S);
    const std::size_t num_out = static_cast<std::size_t>(B * H * S * S);

    std::vector<float> q_data(num_q);
    std::vector<float> k_data(num_k);
    for (std::size_t i = 0; i < num_q; ++i) q_data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    for (std::size_t i = 0; i < num_k; ++i) k_data[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25f;

    q_cpu.copy_from_host(as_bytes(q_data));
    k_cpu.copy_from_host(as_bytes(k_data));
    exec_cpu->execute();
    std::vector<float> ref_out(num_out);
    out_cpu.copy_to_host(as_writeable_bytes(ref_out));

    q_cuda.copy_from_host(as_bytes(q_data));
    k_cuda.copy_from_host(as_bytes(k_data));
    exec_cuda->execute();
    std::vector<float> cuda_out(num_out);
    out_cuda.copy_to_host(as_writeable_bytes(cuda_out));

    REQUIRE(cuda_out.size() == ref_out.size());
    for (std::size_t i = 0; i < num_out; ++i) {
        REQUIRE(std::abs(cuda_out[i] - ref_out[i]) < 1e-4f);
    }
}

TEST_CASE("CUDA MatMul - 3D Batched GEMM equivalence vs CPU", "[cuda][matmul][batched3d]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");

    // [B, S, K] @ [B, K, N] -> [B, S, N] 3D 批量矩阵乘法
    const dim_t B = 3, S = 7, K = 16, N = 24;
    Graph g_cpu;
    auto x_cpu = g_cpu.input({B, S, K}, DataType::Float32);
    auto w_cpu = g_cpu.input({B, K, N}, DataType::Float32);
    auto out_cpu = g_cpu.op(Op::MatMul, x_cpu, w_cpu);
    auto exec_cpu = g_cpu.build(DeviceType::CPU);

    Graph g_cuda;
    auto x_cuda = g_cuda.input({B, S, K}, DataType::Float32);
    auto w_cuda = g_cuda.input({B, K, N}, DataType::Float32);
    auto out_cuda = g_cuda.op(Op::MatMul, x_cuda, w_cuda);
    auto exec_cuda = g_cuda.build(DeviceType::CUDA);

    const std::size_t num_x = static_cast<std::size_t>(B * S * K);
    const std::size_t num_w = static_cast<std::size_t>(B * K * N);
    const std::size_t num_out = static_cast<std::size_t>(B * S * N);

    std::vector<float> x_data(num_x);
    std::vector<float> w_data(num_w);
    for (std::size_t i = 0; i < num_x; ++i) x_data[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.1f;
    for (std::size_t i = 0; i < num_w; ++i) w_data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.05f;

    x_cpu.copy_from_host(as_bytes(x_data));
    w_cpu.copy_from_host(as_bytes(w_data));
    exec_cpu->execute();
    std::vector<float> ref_out(num_out);
    out_cpu.copy_to_host(as_writeable_bytes(ref_out));

    x_cuda.copy_from_host(as_bytes(x_data));
    w_cuda.copy_from_host(as_bytes(w_data));
    exec_cuda->execute();
    std::vector<float> cuda_out(num_out);
    out_cuda.copy_to_host(as_writeable_bytes(cuda_out));

    REQUIRE(cuda_out.size() == ref_out.size());
    for (std::size_t i = 0; i < num_out; ++i) {
        REQUIRE(std::abs(cuda_out[i] - ref_out[i]) < 1e-4f);
    }
}

TEST_CASE("CUDA Boundary Sync & Asynchronous Pipeline Execution", "[cuda][sync]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");

    // 级联算子流水线：输入 -> RMSNorm -> MatMul -> Add -> Relu
    Graph g;
    auto x = g.input({2, 8}, DataType::Float32);
    auto norm_w = g.input({8}, DataType::Float32);
    auto mat_w = g.input({8, 8}, DataType::Float32);
    auto bias = g.input({2, 8}, DataType::Float32);

    auto h = g.op(Op::RMSNorm, x, norm_w);
    auto m = g.op(Op::MatMul, h, mat_w);
    auto a = g.op(Op::Add, m, bias);
    auto y = g.op(Op::Relu, a);

    auto exec = g.build(DeviceType::CUDA);

    std::vector<float> x_data(16, 1.0f);
    std::vector<float> nw_data(8, 1.0f);
    std::vector<float> mw_data(64, 0.1f);
    std::vector<float> b_data(16, -0.5f);

    x.copy_from_host(as_bytes(x_data));
    norm_w.copy_from_host(as_bytes(nw_data));
    mat_w.copy_from_host(as_bytes(mw_data));
    bias.copy_from_host(as_bytes(b_data));

    // 执行并验证边界同步正常收敛
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> out(16);
    y.copy_to_host(as_writeable_bytes(out));
    for (float v : out) {
        REQUIRE(std::abs(v - 0.3f) < 1e-4f);
    }

    // 验证测试注入同步故障能被精准截获
    {
        ScopedBackendFailure failure("CUDA:MatMul:sync");
        try {
            exec->execute();
            FAIL("应报告注入的 MatMul 同步失败");
        } catch (const std::runtime_error& err) {
            REQUIRE(std::string(err.what()).find("MatMul on CUDA output shape=[2,8] at sync") != std::string::npos);
        }
    }
    REQUIRE_NOTHROW(exec->execute());
}

TEST_CASE("CUDA MatMul - Transposed B in Batched GEMM (Attention Q @ K^T)", "[cuda][matmul][transpose]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;
    if (!Device::cuda().is_available()) SKIP("CUDA 设备不可用");

    // 注意力分数量化：Q: [B, H, S, D], K: [B, H, S, D] -> K^T: [B, H, D, S] -> Q @ K^T: [B, H, S, S]
    const dim_t B = 2, H = 2, S = 4, D = 8;
    Graph g;
    auto q = g.input({B, H, S, D}, DataType::Float32);
    auto k = g.input({B, H, S, D}, DataType::Float32);
    auto kt = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {0, 1, 3, 2}}}, k);
    auto scores = g.op(Op::MatMul, q, kt);

    auto exec = g.build(DeviceType::CUDA);

    const std::size_t num_qk = static_cast<std::size_t>(B * H * S * D);
    const std::size_t num_scores = static_cast<std::size_t>(B * H * S * S);

    std::vector<float> q_data(num_qk);
    std::vector<float> k_data(num_qk);
    for (std::size_t i = 0; i < num_qk; ++i) {
        q_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2f;
        k_data[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.1f;
    }

    q.copy_from_host(as_bytes(q_data));
    k.copy_from_host(as_bytes(k_data));
    exec->execute();

    std::vector<float> scores_data(num_scores);
    scores.copy_to_host(as_writeable_bytes(scores_data));

    // 计算 CPU 标量参考
    std::vector<float> ref_scores(num_scores, 0.0f);
    for (dim_t b = 0; b < B; ++b) {
        for (dim_t h = 0; h < H; ++h) {
            for (dim_t s1 = 0; s1 < S; ++s1) {
                for (dim_t s2 = 0; s2 < S; ++s2) {
                    float sum = 0.0f;
                    for (dim_t d = 0; d < D; ++d) {
                        float q_val = q_data[((b * H + h) * S + s1) * D + d];
                        float k_val = k_data[((b * H + h) * S + s2) * D + d];
                        sum += q_val * k_val;
                    }
                    ref_scores[((b * H + h) * S + s1) * S + s2] = sum;
                }
            }
        }
    }

    for (std::size_t i = 0; i < num_scores; ++i) {
        REQUIRE(std::abs(scores_data[i] - ref_scores[i]) < 1e-4f);
    }
}

#endif
