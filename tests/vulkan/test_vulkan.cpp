#include <cmath>
#include <cstdlib>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "../test_helpers.h"

#ifdef VELOMIND_ENABLE_VULKAN

namespace {

    struct ScopedVulkanFailure {
        std::string previous;
        bool had_previous;

        explicit ScopedVulkanFailure(const std::string& spec) {
            const char* value = std::getenv("VELOMIND_FAIL_BACKEND");
            had_previous = value != nullptr;
            if (value) previous = value;
            setenv("VELOMIND_FAIL_BACKEND", spec.c_str(), 1);
        }
        ~ScopedVulkanFailure() {
            if (had_previous) setenv("VELOMIND_FAIL_BACKEND", previous.c_str(), 1);
            else unsetenv("VELOMIND_FAIL_BACKEND");
        }
    };

    template <typename Body>
    void run_or_skip_vulkan(Body&& body) {
        if (!velomind::Device::vulkan().is_available()) {
            SKIP("Vulkan backend not available in this build.");
        }
        body();
    }

}

TEST_CASE("Vulkan errors identify operator stage and output shape", "[vulkan][failure]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");
    Graph g;
    auto a = g.input({2}, DataType::Float32);
    auto b = g.input({2}, DataType::Float32);
    auto y = g.op(Op::Add, a, b);
    auto exec = g.build(DeviceType::VULKAN);
    a.copy_from_host(as_bytes(std::vector<float>{1, 2}));
    b.copy_from_host(as_bytes(std::vector<float>{3, 4}));

    for (const char* stage : {"pipeline", "descriptor allocation", "command allocation",
                              "command begin", "command end", "fence create", "submit",
                              "fence wait", "descriptor reset"}) {
        {
            ScopedVulkanFailure failure(std::string("Vulkan:Add:") + stage);
            try {
                exec->execute();
                FAIL("应报告注入的 Vulkan 失败");
            } catch (const std::runtime_error& error) {
                REQUIRE(std::string(error.what()).find(
                    std::string("Add on Vulkan output shape=[2] at ") + stage) != std::string::npos);
            }
        }
        REQUIRE_NOTHROW(exec->execute());
    }
    std::vector<float> values(2);
    y.copy_to_host(as_writeable_bytes(values));
    REQUIRE(values == std::vector<float>{4, 6});
}

TEST_CASE("Element-wise Add round-trip (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Add, a, b);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

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

TEST_CASE("Element-wise Mul round-trip (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Mul, a, b);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

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

TEST_CASE("Sigmoid round-trip (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4}, DataType::Float32);
    auto y = g.op(Op::Sigmoid, x);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data = {0.0f, 1.0f, -1.0f, 2.0f};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == Catch::Approx(0.5f));
    REQUIRE(y_data[1] == Catch::Approx(1.0f / (1.0f + std::exp(-1.0f))));
    REQUIRE(y_data[2] == Catch::Approx(1.0f / (1.0f + std::exp( 1.0f))));
    REQUIRE(y_data[3] == Catch::Approx(1.0f / (1.0f + std::exp(-2.0f))));
}

TEST_CASE("Embedding gather (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto table   = g.input({4, 3}, DataType::Float32);
    auto indices = g.input({2}, DataType::Int32);
    auto y       = g.op(Op::Embedding, table, indices);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> table_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.f, 11.f, 12.f,
    };
    std::vector<int32_t> idx_data = {2, 0};
    table.copy_from_host(as_bytes(table_data));
    indices.copy_from_host(as_bytes(idx_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == 7.0f);
    REQUIRE(y_data[1] == 8.0f);
    REQUIRE(y_data[2] == 9.0f);

    REQUIRE(y_data[3] == 1.0f);
    REQUIRE(y_data[4] == 2.0f);
    REQUIRE(y_data[5] == 3.0f);
}

TEST_CASE("RMSNorm (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x      = g.input({2, 4}, DataType::Float32);
    auto weight = g.input({4}, DataType::Float32);
    auto y      = g.op(Op::RMSNorm, x, weight);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3, 4,
                                 2, 0, 0, 0};
    std::vector<float> w_data = {1, 1, 1, 1};
    x.copy_from_host(as_bytes(x_data));
    weight.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    const float inv_rms_0 = 1.0f / std::sqrt(7.5f + 1e-5f);
    REQUIRE(y_data[0] == Catch::Approx(1.0f * inv_rms_0));
    REQUIRE(y_data[1] == Catch::Approx(2.0f * inv_rms_0));
    REQUIRE(y_data[2] == Catch::Approx(3.0f * inv_rms_0));
    REQUIRE(y_data[3] == Catch::Approx(4.0f * inv_rms_0));
    REQUIRE(y_data[4] == Catch::Approx(2.0f * 0.99999f).margin(1e-4f));
    REQUIRE(y_data[5] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(y_data[6] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(y_data[7] == Catch::Approx(0.0f).margin(1e-5f));
}

TEST_CASE("RotaryEmbedding rotate-half (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x   = g.input({2, 4}, DataType::Float32);
    auto cos = g.input({2, 2}, DataType::Float32);
    auto sin = g.input({2, 2}, DataType::Float32);
    auto y   = g.op(Op::RotaryEmbedding, x, cos, sin);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data   = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> cos_data = {1, 1, 0, 0};
    std::vector<float> sin_data = {0, 0, 1, 1};
    x.copy_from_host(as_bytes(x_data));
    cos.copy_from_host(as_bytes(cos_data));
    sin.copy_from_host(as_bytes(sin_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == 1.0f);
    REQUIRE(y_data[1] == 2.0f);
    REQUIRE(y_data[2] == 3.0f);
    REQUIRE(y_data[3] == 4.0f);
    REQUIRE(y_data[4] == -7.0f);
    REQUIRE(y_data[5] == -8.0f);
    REQUIRE(y_data[6] == 5.0f);
    REQUIRE(y_data[7] == 6.0f);
}

TEST_CASE("Reshape (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(velomind::OpDescriptor{
                      velomind::Op::Reshape,
                      velomind::ReshapeAttrs{ shape_t{6} }},
                  x);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3,
                                 4, 5, 6};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == 1.0f);
    REQUIRE(y_data[5] == 6.0f);
}

TEST_CASE("Transpose 2x3 -> 3x2 (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(velomind::OpDescriptor{
                      velomind::Op::Transpose,
                      velomind::TransposeAttrs{ std::vector<int>{1, 0} }},
                  x);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3,
                                 4, 5, 6};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == 1.0f);
    REQUIRE(y_data[1] == 4.0f);
    REQUIRE(y_data[2] == 2.0f);
    REQUIRE(y_data[3] == 5.0f);
    REQUIRE(y_data[4] == 3.0f);
    REQUIRE(y_data[5] == 6.0f);
}

TEST_CASE("Softmax row-normalised (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(Op::Softmax, x);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3,
                                 0, 0, 0};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    float e1 = std::exp(1), e2 = std::exp(2), e3 = std::exp(3);
    float s0 = e1 + e2 + e3;
    REQUIRE(y_data[0] == Catch::Approx(e1 / s0).margin(1e-5f));
    REQUIRE(y_data[1] == Catch::Approx(e2 / s0).margin(1e-5f));
    REQUIRE(y_data[2] == Catch::Approx(e3 / s0).margin(1e-5f));

    REQUIRE(y_data[3] == Catch::Approx(1.0f / 3.0f).margin(1e-5f));
    REQUIRE(y_data[4] == Catch::Approx(1.0f / 3.0f).margin(1e-5f));
    REQUIRE(y_data[5] == Catch::Approx(1.0f / 3.0f).margin(1e-5f));
}

TEST_CASE("MatMul 2x3 @ 3x2 (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({2, 3}, DataType::Float32);
    auto b = g.input({3, 2}, DataType::Float32);
    auto c = g.op(Op::MatMul, a, b);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> a_data = {1, 2, 3,
                                 4, 5, 6};
    std::vector<float> b_data = {1, 0,
                                 0, 1,
                                 1, 1};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));

    REQUIRE(c_data[0] == Catch::Approx(4.0f));
    REQUIRE(c_data[1] == Catch::Approx(5.0f));
    REQUIRE(c_data[2] == Catch::Approx(10.0f));
    REQUIRE(c_data[3] == Catch::Approx(11.0f));
}

TEST_CASE("MatMul then Add then Mul (Float32, Vulkan)", "[graph][vulkan]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto gate = g.input({4}, DataType::Float32);
    auto up   = g.input({4}, DataType::Float32);
    auto sig  = g.op(Op::Sigmoid, gate);
    auto silu = g.op(Op::Mul, sig, gate);
    auto y    = g.op(Op::Mul, silu, up);

    auto exec = g.build(DeviceType::VULKAN);
    if (exec == nullptr) SKIP("Vulkan backend not available in this build.");

    std::vector<float> g_data = {0, 1, -1, 2};
    std::vector<float> u_data = {1, 2, 3, 4};
    gate.copy_from_host(as_bytes(g_data));
    up.copy_from_host(as_bytes(u_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    for (int i = 0; i < 4; ++i) {
        float sg = 1.0f / (1.0f + std::exp(-g_data[i]));
        float silu_v = sg * g_data[i];
        REQUIRE(y_data[i] == Catch::Approx(silu_v * u_data[i]).margin(1e-5f));
    }
}

TEST_CASE("Vulkan Transpose - 3D and 4D tensor permutations", "[graph][vulkan][transpose]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");

    // 3D 转置：[2, 3, 4] -> [2, 4, 3]，perm = {0, 2, 1}
    {
        Graph g;
        auto x = g.input({2, 3, 4}, DataType::Float32);
        auto y = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{std::vector<int>{0, 2, 1}}}, x);
        auto exec = g.build(DeviceType::VULKAN);
        REQUIRE(exec != nullptr);

        std::vector<float> in_data(24);
        for (std::size_t i = 0; i < 24; ++i) in_data[i] = static_cast<float>(i + 1);
        x.copy_from_host(as_bytes(in_data));
        exec->execute();

        std::vector<float> out_data(24);
        y.copy_to_host(as_writeable_bytes(out_data));

        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 4; ++c) {
                for (int r = 0; r < 3; ++r) {
                    const std::size_t out_idx = b * 12 + c * 3 + r;
                    const std::size_t in_idx  = b * 12 + r * 4 + c;
                    REQUIRE(out_data[out_idx] == in_data[in_idx]);
                }
            }
        }
    }

    // 4D 转置（多头注意力布局转换）：[2, 3, 4, 5] -> [2, 4, 3, 5]，perm = {0, 2, 1, 3}
    {
        Graph g;
        auto x = g.input({2, 3, 4, 5}, DataType::Float32);
        auto y = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{std::vector<int>{0, 2, 1, 3}}}, x);
        auto exec = g.build(DeviceType::VULKAN);
        REQUIRE(exec != nullptr);

        const std::size_t total = 2 * 3 * 4 * 5;
        std::vector<float> in_data(total);
        for (std::size_t i = 0; i < total; ++i) in_data[i] = static_cast<float>(i * 2 + 1);
        x.copy_from_host(as_bytes(in_data));
        exec->execute();

        std::vector<float> out_data(total);
        y.copy_to_host(as_writeable_bytes(out_data));

        for (int b = 0; b < 2; ++b) {
            for (int h = 0; h < 4; ++h) {
                for (int s = 0; s < 3; ++s) {
                    for (int d = 0; d < 5; ++d) {
                        const std::size_t out_idx = ((b * 4 + h) * 3 + s) * 5 + d;
                        const std::size_t in_idx  = ((b * 3 + s) * 4 + h) * 5 + d;
                        REQUIRE(out_data[out_idx] == in_data[in_idx]);
                    }
                }
            }
        }
    }
}

TEST_CASE("Vulkan Concurrent Multi-Thread Execution", "[vulkan][concurrency]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");

    constexpr int NUM_THREADS = 4;
    std::vector<std::future<bool>> futures;

    for (int t = 0; t < NUM_THREADS; ++t) {
        futures.push_back(std::async(std::launch::async, [t]() -> bool {
            Graph g;
            auto a = g.input({8}, DataType::Float32);
            auto b = g.input({8}, DataType::Float32);
            auto c = g.op(Op::Add, a, b);
            auto d = g.op(Op::Mul, c, a);
            auto exec = g.build(DeviceType::VULKAN);
            if (!exec) return false;

            std::vector<float> a_data(8, static_cast<float>(t + 1));
            std::vector<float> b_data(8, 10.0f);
            a.copy_from_host(as_bytes(a_data));
            b.copy_from_host(as_bytes(b_data));

            for (int iter = 0; iter < 10; ++iter) {
                exec->execute();
            }

            std::vector<float> d_data(8);
            d.copy_to_host(as_writeable_bytes(d_data));

            const float expected = (static_cast<float>(t + 1) + 10.0f) * static_cast<float>(t + 1);
            for (float val : d_data) {
                if (std::abs(val - expected) > 1e-4f) return false;
            }
            return true;
        }));
    }

    for (auto& f : futures) {
        REQUIRE(f.get() == true);
    }
}

TEST_CASE("Vulkan Batch Command Recording with Complex Graph", "[graph][vulkan][batch]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");

    Graph g;
    auto x    = g.input({2, 4}, DataType::Float32);
    auto w    = g.input({4}, DataType::Float32);
    auto bias = g.input({2, 4}, DataType::Float32);

    auto norm = g.op(Op::RMSNorm, x, w);
    auto add  = g.op(Op::Add, norm, bias);
    auto sig  = g.op(Op::Sigmoid, add);
    auto mul  = g.op(Op::Mul, add, sig);
    auto tr   = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{std::vector<int>{1, 0}}}, mul);
    auto sm   = g.op(Op::Softmax, tr);

    auto exec = g.build(DeviceType::VULKAN);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1, 2, 3, 4, 2, 0, 0, 0};
    std::vector<float> w_data = {1, 1, 1, 1};
    std::vector<float> b_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    x.copy_from_host(as_bytes(x_data));
    w.copy_from_host(as_bytes(w_data));
    bias.copy_from_host(as_bytes(b_data));

    exec->execute();

    std::vector<float> sm_data(sm.numel());
    sm.copy_to_host(as_writeable_bytes(sm_data));

    // 验证 Softmax 每行概率和为 1.0 (输出形状为 [4, 2])
    for (int r = 0; r < 4; ++r) {
        float sum = sm_data[r * 2 + 0] + sm_data[r * 2 + 1];
        REQUIRE(sum == Catch::Approx(1.0f).margin(1e-4f));
    }
}

TEST_CASE("Vulkan Op::Concat 1-D and 2-D", "[vulkan][concat]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    if (!Device::vulkan().is_available()) SKIP("Vulkan 设备不可用");

    Graph g;
    auto a = g.input({2, 3}, DataType::Float32);
    auto b = g.input({2, 2}, DataType::Float32);
    auto c = g.op(OpDescriptor{Op::Concat, ConcatAttrs{1}}, a, b);

    auto exec = g.build(DeviceType::VULKAN);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f};

    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));

    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));

    std::vector<float> expected = {
        1.0f, 2.0f, 3.0f, 7.0f, 8.0f,
        4.0f, 5.0f, 6.0f, 9.0f, 10.0f
    };
    REQUIRE(c_data == expected);
}

#endif
