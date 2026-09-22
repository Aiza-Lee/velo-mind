#include <cmath>
#include <vector>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "velomind/device.h"
#include "velomind/dtype.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "test_model_assets.h"

TEST_CASE("float16_t conversion and bit-exact representation", "[dtype][fp16]") {
    using namespace velomind;

    // 常用精确值与 IEEE 754 半精度位模式对齐测试
    CHECK(float16_t(0.0f).to_bits() == 0x0000);
    CHECK(float16_t(-0.0f).to_bits() == 0x8000);
    CHECK(float16_t(1.0f).to_bits() == 0x3C00);
    CHECK(float16_t(-2.0f).to_bits() == 0xC000);
    CHECK(float16_t(0.5f).to_bits() == 0x3800);
    CHECK(float16_t(65504.0f).to_bits() == 0x7BFF);

    // 逆向解包与转换回 float 的数值等价性
    CHECK(static_cast<float>(float16_t::from_bits(0x3C00)) == Catch::Approx(1.0f));
    CHECK(static_cast<float>(float16_t::from_bits(0xC000)) == Catch::Approx(-2.0f));
    CHECK(static_cast<float>(float16_t::from_bits(0x3800)) == Catch::Approx(0.5f));
    CHECK(static_cast<float>(float16_t::from_bits(0x7BFF)) == Catch::Approx(65504.0f));

    // 特殊值：Inf 与 NaN
    float inf_f = std::numeric_limits<float>::infinity();
    float16_t inf_h(inf_f);
    CHECK(inf_h.to_bits() == 0x7C00);
    CHECK(std::isinf(static_cast<float>(inf_h)));

    float neg_inf_f = -std::numeric_limits<float>::infinity();
    float16_t neg_inf_h(neg_inf_f);
    CHECK(neg_inf_h.to_bits() == 0xFC00);
    CHECK(std::isinf(static_cast<float>(neg_inf_h)));

    float nan_f = std::numeric_limits<float>::quiet_NaN();
    float16_t nan_h(nan_f);
    CHECK(std::isnan(static_cast<float>(nan_h)));

    // 比较运算符
    float16_t a(1.5f);
    float16_t b(1.5f);
    float16_t c(2.5f);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("bfloat16_t conversion and bit-exact representation", "[dtype][bf16]") {
    using namespace velomind;

    // Google Brain 16 位浮点数高 16 位截断与对齐验证
    CHECK(bfloat16_t(0.0f).to_bits() == 0x0000);
    CHECK(bfloat16_t(1.0f).to_bits() == 0x3F80);
    CHECK(bfloat16_t(-2.0f).to_bits() == 0xC000);
    CHECK(bfloat16_t(0.5f).to_bits() == 0x3F00);

    CHECK(static_cast<float>(bfloat16_t::from_bits(0x3F80)) == Catch::Approx(1.0f));
    CHECK(static_cast<float>(bfloat16_t::from_bits(0xC000)) == Catch::Approx(-2.0f));
    CHECK(static_cast<float>(bfloat16_t::from_bits(0x3F00)) == Catch::Approx(0.5f));

    // 特殊值测试
    float inf_f = std::numeric_limits<float>::infinity();
    bfloat16_t inf_b(inf_f);
    CHECK(std::isinf(static_cast<float>(inf_b)));

    float nan_f = std::numeric_limits<float>::quiet_NaN();
    bfloat16_t nan_b(nan_f);
    CHECK(std::isnan(static_cast<float>(nan_b)));

    bfloat16_t a(3.14159f);
    bfloat16_t b(3.14159f);
    bfloat16_t c(2.71828f);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("convert_dtype buffer conversions across precision formats", "[dtype][convert]") {
    using namespace velomind;

    constexpr std::size_t n = 6;
    std::vector<float> orig_f32 = { 0.0f, -1.0f, 0.5f, 2.0f, 10.5f, -3.25f };

    // Float32 经 Float16 往返转换
    std::vector<float16_t> buf_f16(n);
    convert_dtype(orig_f32.data(), DataType::Float32, buf_f16.data(), DataType::Float16, n);

    std::vector<float> back_f32(n);
    convert_dtype(buf_f16.data(), DataType::Float16, back_f32.data(), DataType::Float32, n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(back_f32[i] == Catch::Approx(orig_f32[i]).margin(1e-3f));
    }

    // Float32 经 BFloat16 往返转换
    std::vector<bfloat16_t> buf_bf16(n);
    convert_dtype(orig_f32.data(), DataType::Float32, buf_bf16.data(), DataType::BFloat16, n);

    std::vector<float> back_from_bf16(n);
    convert_dtype(buf_bf16.data(), DataType::BFloat16, back_from_bf16.data(), DataType::Float32, n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(back_from_bf16[i] == Catch::Approx(orig_f32[i]).margin(0.05f));
    }

    // Float16 -> BFloat16 跨半精度转换
    std::vector<bfloat16_t> f16_to_bf16(n);
    convert_dtype(buf_f16.data(), DataType::Float16, f16_to_bf16.data(), DataType::BFloat16, n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(static_cast<float>(f16_to_bf16[i]) == Catch::Approx(orig_f32[i]).margin(0.05f));
    }

    // 整型 -> Float16
    std::vector<std::int32_t> ints = { 0, 1, -5, 42, 100, -128 };
    std::vector<float16_t> ints_f16(n);
    convert_dtype(ints.data(), DataType::Int32, ints_f16.data(), DataType::Float16, n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(static_cast<float>(ints_f16[i]) == Catch::Approx(static_cast<float>(ints[i])));
    }
}

TEST_CASE("dtype capability queries is_dtype_supported and is_op_dtype_supported", "[dtype][query]") {
    using namespace velomind;

    CHECK(is_dtype_supported(DeviceType::CPU, DataType::Float32));
    CHECK(is_dtype_supported(DeviceType::CPU, DataType::Float16));
    CHECK(is_dtype_supported(DeviceType::CPU, DataType::BFloat16));

    // 检查 CPU 算子派发支持
    std::vector<DataType> binary_f16 = { DataType::Float16, DataType::Float16 };
    CHECK(is_op_dtype_supported(DeviceType::CPU, Op::MatMul, binary_f16, DataType::Float16));
    CHECK(is_op_dtype_supported(DeviceType::CPU, Op::MatMul, binary_f16, DataType::Float32));

    std::vector<DataType> mixed_f32_f16 = { DataType::Float32, DataType::Float16 };
    CHECK(is_op_dtype_supported(DeviceType::CPU, Op::MatMul, mixed_f32_f16, DataType::Float32));

    std::vector<DataType> unary_f16 = { DataType::Float16 };
    CHECK(is_op_dtype_supported(DeviceType::CPU, Op::Softmax, unary_f16, DataType::Float16));

    if (Device::cuda().is_available()) {
        CHECK(is_dtype_supported(DeviceType::CUDA, DataType::Float16));
        CHECK(is_dtype_supported(DeviceType::CUDA, DataType::BFloat16));
        CHECK(is_op_dtype_supported(DeviceType::CUDA, Op::MatMul, binary_f16, DataType::Float16));
        CHECK(is_op_dtype_supported(DeviceType::CUDA, Op::MatMul, mixed_f32_f16, DataType::Float32));
    }
}

TEST_CASE("FP16 and BF16 MatMul inference and FP32 accumulation on CPU", "[dtype][matmul][cpu]") {
    using namespace velomind;

    constexpr dim_t M = 4;
    constexpr dim_t K = 8;
    constexpr dim_t N = 6;

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    for (std::size_t i = 0; i < M * K; ++i) h_a[i] = static_cast<float>(i + 1) * 0.1f;
    for (std::size_t i = 0; i < K * N; ++i) h_b[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;

    // 计算 FP32 黄金参考结果
    std::vector<float> golden(M * N, 0.0f);
    for (dim_t m = 0; m < M; ++m) {
        for (dim_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (dim_t k = 0; k < K; ++k) {
                sum += h_a[m * K + k] * h_b[k * N + n];
            }
            golden[m * N + n] = sum;
        }
    }
    velomind_test::require_all_finite(golden, "golden");

    // 同精度 FP16 MatMul: (Float16, Float16) -> Float16
    {
        Graph g;
        auto a = g.input({M, K}, DataType::Float16);
        auto b = g.input({K, N}, DataType::Float16);
        auto c = g.op(Op::MatMul, a, b);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        std::vector<float16_t> a_f16(M * K);
        std::vector<float16_t> b_f16(K * N);
        convert_dtype(h_a.data(), DataType::Float32, a_f16.data(), DataType::Float16, M * K);
        convert_dtype(h_b.data(), DataType::Float32, b_f16.data(), DataType::Float16, K * N);

        a.copy_from_host(std::as_bytes(std::span(a_f16)));
        b.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec->execute();

        std::vector<float16_t> c_f16(M * N);
        c.copy_to_host(std::as_writable_bytes(std::span(c_f16)));

        std::vector<float> actual(M * N);
        convert_dtype(c_f16.data(), DataType::Float16, actual.data(), DataType::Float32, M * N);
        velomind_test::require_all_finite(actual, "c_f16");

        for (std::size_t i = 0; i < M * N; ++i) {
            CHECK(actual[i] == Catch::Approx(golden[i]).margin(0.05f));
        }
    }

    // 混合精度 MatMul: (Float32, Float16) -> Float32（低精度权重存储，FP32 累加与输出）
    {
        Graph g;
        auto a = g.input({M, K}, DataType::Float32);
        auto b = g.input({K, N}, DataType::Float16);
        auto c = g.op(Op::MatMul, a, b);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        std::vector<float16_t> b_f16(K * N);
        convert_dtype(h_b.data(), DataType::Float32, b_f16.data(), DataType::Float16, K * N);

        a.copy_from_host(std::as_bytes(std::span(h_a)));
        b.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec->execute();

        std::vector<float> actual(M * N);
        c.copy_to_host(std::as_writable_bytes(std::span(actual)));

        velomind_test::require_all_finite(actual, "mixed_f32_f16");
        for (std::size_t i = 0; i < M * N; ++i) {
            CHECK(actual[i] == Catch::Approx(golden[i]).margin(0.05f));
        }
    }

    // 混合精度 MatMul: (Float32, BFloat16) -> Float32
    {
        Graph g;
        auto a = g.input({M, K}, DataType::Float32);
        auto b = g.input({K, N}, DataType::BFloat16);
        auto c = g.op(Op::MatMul, a, b);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        std::vector<bfloat16_t> b_bf16(K * N);
        convert_dtype(h_b.data(), DataType::Float32, b_bf16.data(), DataType::BFloat16, K * N);

        a.copy_from_host(std::as_bytes(std::span(h_a)));
        b.copy_from_host(std::as_bytes(std::span(b_bf16)));
        exec->execute();

        std::vector<float> actual(M * N);
        c.copy_to_host(std::as_writable_bytes(std::span(actual)));

        velomind_test::require_all_finite(actual, "mixed_f32_bf16");
        for (std::size_t i = 0; i < M * N; ++i) {
            CHECK(actual[i] == Catch::Approx(golden[i]).margin(0.1f));
        }
    }
}

TEST_CASE("FP16 and BF16 RMSNorm and Softmax with FP32 accumulation on CPU", "[dtype][norm][cpu]") {
    using namespace velomind;

    constexpr dim_t batch = 2;
    constexpr dim_t dim = 8;

    std::vector<float> x_f32 = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 1.5f,
        0.1f, -0.5f, 2.2f, -1.1f, 0.9f, 3.3f, -2.2f, 1.1f
    };
    std::vector<float> w_f32 = { 1.0f, 0.5f, 2.0f, 1.5f, 0.8f, 1.2f, 1.0f, 0.9f };

    // RMSNorm 混合精度与 FP32 累加
    {
        Graph g;
        auto x = g.input({batch, dim}, DataType::Float32);
        auto w = g.input({dim}, DataType::Float16);
        auto y = g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 1e-5f, .axis = -1}}, x, w);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        std::vector<float16_t> w_f16(dim);
        convert_dtype(w_f32.data(), DataType::Float32, w_f16.data(), DataType::Float16, dim);

        x.copy_from_host(std::as_bytes(std::span(x_f32)));
        w.copy_from_host(std::as_bytes(std::span(w_f16)));
        exec->execute();

        std::vector<float> actual(batch * dim);
        y.copy_to_host(std::as_writable_bytes(std::span(actual)));

        velomind_test::require_all_finite(actual, "rmsnorm_mixed");
        CHECK(actual[0] > 0.0f);
    }

    // Softmax FP16
    {
        Graph g;
        auto x = g.input({batch, dim}, DataType::Float16);
        auto y = g.op(OpDescriptor{Op::Softmax, SoftmaxAttrs{.axis = -1}}, x);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec != nullptr);

        std::vector<float16_t> x_f16(batch * dim);
        convert_dtype(x_f32.data(), DataType::Float32, x_f16.data(), DataType::Float16, batch * dim);

        x.copy_from_host(std::as_bytes(std::span(x_f16)));
        exec->execute();

        std::vector<float16_t> y_f16(batch * dim);
        y.copy_to_host(std::as_writable_bytes(std::span(y_f16)));

        std::vector<float> actual(batch * dim);
        convert_dtype(y_f16.data(), DataType::Float16, actual.data(), DataType::Float32, batch * dim);
        velomind_test::require_all_finite(actual, "softmax_f16");

        // 验证每行概率和为 1.0
        for (dim_t b = 0; b < batch; ++b) {
            float row_sum = 0.0f;
            for (dim_t d = 0; d < dim; ++d) {
                float prob = actual[b * dim + d];
                CHECK(prob >= 0.0f);
                row_sum += prob;
            }
            CHECK(row_sum == Catch::Approx(1.0f).margin(0.01f));
        }
    }
}

TEST_CASE("FP16 and BF16 cross-backend execution on CUDA vs CPU reference", "[dtype][cuda]") {
    using namespace velomind;

    if (!Device::cuda().is_available()) {
        SKIP("CUDA 后端不可用，跳过跨后端验收");
    }

    constexpr dim_t M = 4;
    constexpr dim_t K = 8;
    constexpr dim_t N = 6;

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    for (std::size_t i = 0; i < M * K; ++i) h_a[i] = static_cast<float>(i + 1) * 0.1f;
    for (std::size_t i = 0; i < K * N; ++i) h_b[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;

    // 同精度 FP16 MatMul: CPU 与 CUDA 对齐
    {
        std::vector<float16_t> a_f16(M * K);
        std::vector<float16_t> b_f16(K * N);
        convert_dtype(h_a.data(), DataType::Float32, a_f16.data(), DataType::Float16, M * K);
        convert_dtype(h_b.data(), DataType::Float32, b_f16.data(), DataType::Float16, K * N);

        // CPU 执行
        Graph g_cpu;
        auto ca = g_cpu.input({M, K}, DataType::Float16);
        auto cb = g_cpu.input({K, N}, DataType::Float16);
        auto cc = g_cpu.op(Op::MatMul, ca, cb);
        auto exec_cpu = g_cpu.build(DeviceType::CPU);

        ca.copy_from_host(std::as_bytes(std::span(a_f16)));
        cb.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec_cpu->execute();

        std::vector<float16_t> cpu_out(M * N);
        cc.copy_to_host(std::as_writable_bytes(std::span(cpu_out)));

        // CUDA 执行
        Graph g_cuda;
        auto ga = g_cuda.input({M, K}, DataType::Float16);
        auto gb = g_cuda.input({K, N}, DataType::Float16);
        auto gc = g_cuda.op(Op::MatMul, ga, gb);
        auto exec_cuda = g_cuda.build(DeviceType::CUDA);

        ga.copy_from_host(std::as_bytes(std::span(a_f16)));
        gb.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec_cuda->execute();

        std::vector<float16_t> cuda_out(M * N);
        gc.copy_to_host(std::as_writable_bytes(std::span(cuda_out)));

        std::vector<float> cpu_f32(M * N);
        std::vector<float> cuda_f32(M * N);
        convert_dtype(cpu_out.data(), DataType::Float16, cpu_f32.data(), DataType::Float32, M * N);
        convert_dtype(cuda_out.data(), DataType::Float16, cuda_f32.data(), DataType::Float32, M * N);

        velomind_test::require_all_finite(cpu_f32, "cpu_f16_matmul");
        velomind_test::require_all_finite(cuda_f32, "cuda_f16_matmul");

        for (std::size_t i = 0; i < M * N; ++i) {
            CHECK(cuda_f32[i] == Catch::Approx(cpu_f32[i]).margin(0.02f));
        }
    }

    // 混合精度 (Float32, Float16) -> Float32: CPU 与 CUDA 对齐
    {
        std::vector<float16_t> b_f16(K * N);
        convert_dtype(h_b.data(), DataType::Float32, b_f16.data(), DataType::Float16, K * N);

        // CPU 执行
        Graph g_cpu;
        auto ca = g_cpu.input({M, K}, DataType::Float32);
        auto cb = g_cpu.input({K, N}, DataType::Float16);
        auto cc = g_cpu.op(Op::MatMul, ca, cb);
        auto exec_cpu = g_cpu.build(DeviceType::CPU);

        ca.copy_from_host(std::as_bytes(std::span(h_a)));
        cb.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec_cpu->execute();

        std::vector<float> cpu_out(M * N);
        cc.copy_to_host(std::as_writable_bytes(std::span(cpu_out)));

        // CUDA 执行
        Graph g_cuda;
        auto ga = g_cuda.input({M, K}, DataType::Float32);
        auto gb = g_cuda.input({K, N}, DataType::Float16);
        auto gc = g_cuda.op(Op::MatMul, ga, gb);
        auto exec_cuda = g_cuda.build(DeviceType::CUDA);

        ga.copy_from_host(std::as_bytes(std::span(h_a)));
        gb.copy_from_host(std::as_bytes(std::span(b_f16)));
        exec_cuda->execute();

        std::vector<float> cuda_out(M * N);
        gc.copy_to_host(std::as_writable_bytes(std::span(cuda_out)));

        velomind_test::require_all_finite(cpu_out, "cpu_mixed_matmul");
        velomind_test::require_all_finite(cuda_out, "cuda_mixed_matmul");

        for (std::size_t i = 0; i < M * N; ++i) {
            CHECK(cuda_out[i] == Catch::Approx(cpu_out[i]).margin(0.01f));
        }
    }

    // RMSNorm: CPU 与 CUDA 对齐
    {
        constexpr dim_t dim = 8;
        std::vector<float> x_f32 = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 1.5f };
        std::vector<float16_t> w_f16(dim);
        std::vector<float> w_f32 = { 1.0f, 0.5f, 2.0f, 1.5f, 0.8f, 1.2f, 1.0f, 0.9f };
        convert_dtype(w_f32.data(), DataType::Float32, w_f16.data(), DataType::Float16, dim);

        Graph g_cpu;
        auto ca = g_cpu.input({1, dim}, DataType::Float32);
        auto cb = g_cpu.input({dim}, DataType::Float16);
        auto cc = g_cpu.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 1e-5f, .axis = -1}}, ca, cb);
        auto exec_cpu = g_cpu.build(DeviceType::CPU);

        ca.copy_from_host(std::as_bytes(std::span(x_f32)));
        cb.copy_from_host(std::as_bytes(std::span(w_f16)));
        exec_cpu->execute();

        std::vector<float> cpu_out(dim);
        cc.copy_to_host(std::as_writable_bytes(std::span(cpu_out)));

        Graph g_cuda;
        auto ga = g_cuda.input({1, dim}, DataType::Float32);
        auto gb = g_cuda.input({dim}, DataType::Float16);
        auto gc = g_cuda.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 1e-5f, .axis = -1}}, ga, gb);
        auto exec_cuda = g_cuda.build(DeviceType::CUDA);

        ga.copy_from_host(std::as_bytes(std::span(x_f32)));
        gb.copy_from_host(std::as_bytes(std::span(w_f16)));
        exec_cuda->execute();

        std::vector<float> cuda_out(dim);
        gc.copy_to_host(std::as_writable_bytes(std::span(cuda_out)));

        velomind_test::require_all_finite(cpu_out, "rmsnorm_cpu");
        velomind_test::require_all_finite(cuda_out, "rmsnorm_cuda");

        for (std::size_t i = 0; i < dim; ++i) {
            CHECK(cuda_out[i] == Catch::Approx(cpu_out[i]).margin(0.01f));
        }
    }
}
