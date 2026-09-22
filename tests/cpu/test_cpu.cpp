#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "../test_helpers.h"

TEST_CASE("Element-wise Add round-trip (Float32)", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Add, a, b);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);
    REQUIRE(exec->num_nodes() == 1);

    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};
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

TEST_CASE("MatMul then Add then Relu (Float32)", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x  = g.input({2, 3}, DataType::Float32);
    auto W  = g.input({3, 2}, DataType::Float32);
    auto b  = g.input({2, 2}, DataType::Float32);
    auto mm  = g.op(Op::MatMul, x, W);
    auto pre = g.op(Op::Add, mm, b);
    auto y   = g.op(Op::Relu, pre);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1, 2, 3,
                                  4, 5, 6};
    std::vector<float> W_data = {1, 0,
                                  0, 1,
                                  1, 1};
    std::vector<float> b_data = {0, 0, 0, 0};
    x.copy_from_host(as_bytes(x_data));
    W.copy_from_host(as_bytes(W_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == 4.0f);
    REQUIRE(y_data[1] == 5.0f);
    REQUIRE(y_data[2] == 10.0f);
    REQUIRE(y_data[3] == 11.0f);
}

TEST_CASE("Uniform-precision MatMul (Float32)", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 2}, DataType::Float32);
    auto W = g.input({2, 2}, DataType::Float32);
    auto y = g.op(Op::MatMul, x, W);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 2.0f,
                                  3.0f, 4.0f};
    std::vector<float> W_data = {1.0f, 0.0f,
                                  0.0f, 1.0f};
    x.copy_from_host(as_bytes(x_data));
    W.copy_from_host(as_bytes(W_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == 1.0f);
    REQUIRE(y_data[1] == 2.0f);
    REQUIRE(y_data[2] == 3.0f);
    REQUIRE(y_data[3] == 4.0f);
}

TEST_CASE("MatMul non-square (Float32)", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto W = g.input({2, 3}, DataType::Float32);
    auto x = g.input({3, 2}, DataType::Float32);
    auto y = g.op(Op::MatMul, W, x);

    REQUIRE(y.dtype() == DataType::Float32);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> W_data = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    std::vector<float>  x_data = {7.0f,  8.0f,
                                  9.0f,  10.0f,
                                  11.0f, 12.0f};
    W.copy_from_host(as_bytes(W_data));
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == Catch::Approx(58.0f));
    REQUIRE(y_data[1] == Catch::Approx(64.0f));
    REQUIRE(y_data[2] == Catch::Approx(139.0f));
    REQUIRE(y_data[3] == Catch::Approx(154.0f));
}

TEST_CASE("MatMul batched (Float32)", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;

    auto a = g.input({2, 2, 3}, DataType::Float32);
    auto b = g.input({2, 3, 2}, DataType::Float32);
    auto c = g.op(Op::MatMul, a, b);
    REQUIRE(c.shape().size() == 3);
    REQUIRE(c.shape()[0] == 2);
    REQUIRE(c.shape()[1] == 2);
    REQUIRE(c.shape()[2] == 2);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data = {1, 2, 3,
                                  4, 5, 6,
                                  2, 0, 1,
                                  1, 1, 1};
    std::vector<float> b_data = {1, 0,
                                  0, 1,
                                  1, 1,
                                  1, 1,
                                  1, 1,
                                  0, 1};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> c_data(c.numel());
    c.copy_to_host(as_writeable_bytes(c_data));

    REQUIRE(c_data[0] == Catch::Approx(4.0f));
    REQUIRE(c_data[1] == Catch::Approx(5.0f));

    REQUIRE(c_data[2] == Catch::Approx(10.0f));
    REQUIRE(c_data[3] == Catch::Approx(11.0f));

    REQUIRE(c_data[4] == Catch::Approx(2.0f));
    REQUIRE(c_data[5] == Catch::Approx(3.0f));

    REQUIRE(c_data[6] == Catch::Approx(2.0f));
    REQUIRE(c_data[7] == Catch::Approx(3.0f));
}

TEST_CASE("CPU op coverage: Abs / Div / Sigmoid / Tanh / Softmax", "[graph][cpu]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    auto build_for_cpu = [](Graph& g) {
        return g.build(DeviceType::CPU);
    };

    {
        Graph g;
        auto x = g.input({4}, DataType::Float32);
        auto y = g.op(Op::Abs, x);
        auto exec = build_for_cpu(g);
        REQUIRE(exec != nullptr);
        std::vector<float> x_data = {-2.5f, -0.5f, 0.5f, 2.5f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == 2.5f);
        REQUIRE(y_data[1] == 0.5f);
        REQUIRE(y_data[2] == 0.5f);
        REQUIRE(y_data[3] == 2.5f);
    }

    {
        Graph g;
        auto a = g.input({4}, DataType::Float32);
        auto b = g.input({4}, DataType::Float32);
        auto c = g.op(Op::Div, a, b);
        auto exec = build_for_cpu(g);
        REQUIRE(exec != nullptr);
        std::vector<float> a_data = {10, 20, 30, 40};
        std::vector<float> b_data = {2, 4, 5, 8};
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();
        std::vector<float> c_data(c.numel());
        c.copy_to_host(as_writeable_bytes(c_data));
        REQUIRE(c_data[0] == 5.0f);
        REQUIRE(c_data[3] == 5.0f);
    }

    {
        Graph g;
        auto x = g.input({3}, DataType::Float32);
        auto y = g.op(Op::Sigmoid, x);
        auto exec = build_for_cpu(g);
        REQUIRE(exec != nullptr);
        std::vector<float> x_data = {0.0f, 1.0f, -1.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == Catch::Approx(0.5f));
        REQUIRE(y_data[1] == Catch::Approx(0.7310586f));
        REQUIRE(y_data[2] == Catch::Approx(0.2689414f));
    }

    {
        Graph g;
        auto x = g.input({3}, DataType::Float32);
        auto y = g.op(Op::Tanh, x);
        auto exec = build_for_cpu(g);
        REQUIRE(exec != nullptr);
        std::vector<float> x_data = {0.0f, 1.0f, -1.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == Catch::Approx(0.0f));
        REQUIRE(y_data[1] == Catch::Approx(0.7615942f));
        REQUIRE(y_data[2] == Catch::Approx(-0.7615942f));
    }

    {
        Graph g;
        auto x = g.input({2, 3}, DataType::Float32);
        auto y = g.op(Op::Softmax, x);
        auto exec = build_for_cpu(g);
        REQUIRE(exec != nullptr);
        std::vector<float> x_data = {1.0f, 2.0f, 3.0f,
                                      -1.0f, 0.0f, 1.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        float row0_sum = y_data[0] + y_data[1] + y_data[2];
        float row1_sum = y_data[3] + y_data[4] + y_data[5];
        REQUIRE(row0_sum == Catch::Approx(1.0f));
        REQUIRE(row1_sum == Catch::Approx(1.0f));
    }
}

TEST_CASE("Op::Rsqrt round-trip (Float32)", "[graph][cpu][rsqrt]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4}, DataType::Float32);
    auto y = g.op(Op::Rsqrt, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 4.0f, 9.0f, 16.0f};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data[0] == Catch::Approx(1.0f));
    REQUIRE(y_data[1] == Catch::Approx(0.5f));
    REQUIRE(y_data[2] == Catch::Approx(1.0f / 3.0f));
    REQUIRE(y_data[3] == Catch::Approx(0.25f));
}

TEST_CASE("Op::RMSNorm default eps (Float32)", "[graph][cpu][rmsnorm]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x      = g.input({1, 4}, DataType::Float32);
    auto weight = g.input({4},    DataType::Float32);

    auto y      = g.op(Op::RMSNorm, x, weight);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_data = {1.0f, 1.0f, 1.0f, 1.0f};
    x.copy_from_host(as_bytes(x_data));
    weight.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    const float inv_rms = 1.0f / std::sqrt(7.5f + 1e-5f);
    REQUIRE(y_data[0] == Catch::Approx(1.0f * inv_rms));
    REQUIRE(y_data[1] == Catch::Approx(2.0f * inv_rms));
    REQUIRE(y_data[2] == Catch::Approx(3.0f * inv_rms));
    REQUIRE(y_data[3] == Catch::Approx(4.0f * inv_rms));
}

TEST_CASE("Op::RMSNorm with non-trivial weight (Float32)", "[graph][cpu][rmsnorm]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x      = g.input({2, 4}, DataType::Float32);
    auto weight = g.input({4},     DataType::Float32);
    auto y      = g.op(Op::RMSNorm, x, weight);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                  2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> w_data = {2.0f, 2.0f, 2.0f, 2.0f};
    x.copy_from_host(as_bytes(x_data));
    weight.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    const float r0 = 1.0f / std::sqrt(7.5f + 1e-5f);
    const float r1 = 1.0f / std::sqrt(4.0f + 1e-5f);
    REQUIRE(y_data[0] == Catch::Approx(2.0f * 1.0f * r0));
    REQUIRE(y_data[1] == Catch::Approx(2.0f * 2.0f * r0));
    REQUIRE(y_data[2] == Catch::Approx(2.0f * 3.0f * r0));
    REQUIRE(y_data[3] == Catch::Approx(2.0f * 4.0f * r0));
    REQUIRE(y_data[4] == Catch::Approx(2.0f * 2.0f * r1));
    REQUIRE(y_data[5] == Catch::Approx(2.0f * 2.0f * r1));
    REQUIRE(y_data[6] == Catch::Approx(2.0f * 2.0f * r1));
    REQUIRE(y_data[7] == Catch::Approx(2.0f * 2.0f * r1));
}

TEST_CASE("Op::RMSNorm with custom eps via OpDescriptor (Float32)", "[graph][cpu][rmsnorm]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x      = g.input({1, 4}, DataType::Float32);
    auto weight = g.input({4},    DataType::Float32);

    auto y = g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{1e-2f, -1}},
                  x, weight);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_data = {1.0f, 1.0f, 1.0f, 1.0f};
    x.copy_from_host(as_bytes(x_data));
    weight.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    const float inv_rms = 1.0f / std::sqrt(7.5f + 1e-2f);
    REQUIRE(y_data[0] == Catch::Approx(1.0f * inv_rms));
    REQUIRE(y_data[3] == Catch::Approx(4.0f * inv_rms));
}

TEST_CASE("Op::Embedding gather (Float32 table, Int32 indices)",
          "[graph][cpu][embedding]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto table   = g.input({4, 3}, DataType::Float32);
    auto indices = g.input({3},    DataType::Int32);
    auto y       = g.op(Op::Embedding, table, indices);
    auto exec    = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float>   t_data = { 1,  2,  3,
                                     4,  5,  6,
                                     7,  8,  9,
                                    10, 11, 12};

    std::vector<int32_t> i_data = {0, 2, 3};
    table.copy_from_host(as_bytes(t_data));
    indices.copy_from_host(as_bytes(i_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data == std::vector<float>{1, 2, 3,
                                          7, 8, 9,
                                         10,11,12});
}

TEST_CASE("Op::Embedding gather (2-D indices)", "[graph][cpu][embedding]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto table   = g.input({3, 2}, DataType::Float32);
    auto indices = g.input({2, 2}, DataType::Int32);
    auto y       = g.op(Op::Embedding, table, indices);
    auto exec    = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float>   t_data = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> i_data = {0, 2,
                                    1, 0};
    table.copy_from_host(as_bytes(t_data));
    indices.copy_from_host(as_bytes(i_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.numel()      == 8);
    REQUIRE(y_data.size()  == 8);

    REQUIRE(y_data == std::vector<float>{1, 2, 5, 6,
                                          3, 4, 1, 2});
}

TEST_CASE("Op::RotaryEmbedding rotate-half (Float32)",
          "[graph][cpu][rope]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x   = g.input({4}, DataType::Float32);
    auto cos = g.input({2}, DataType::Float32);
    auto sin = g.input({2}, DataType::Float32);
    auto y   = g.op(Op::RotaryEmbedding, x, cos, sin);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data   = {1, 2, 3, 4};
    std::vector<float> cos_data = {1, 0};
    std::vector<float> sin_data = {0, 1};
    x.copy_from_host(as_bytes(x_data));
    cos.copy_from_host(as_bytes(cos_data));
    sin.copy_from_host(as_bytes(sin_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data == std::vector<float>{1, -4, 3, 2});
}

TEST_CASE("Op::RotaryEmbedding identity when cos=1, sin=0 (Float32)",
          "[graph][cpu][rope]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x   = g.input({6}, DataType::Float32);
    auto cos = g.input({3}, DataType::Float32);
    auto sin = g.input({3}, DataType::Float32);
    auto y   = g.op(Op::RotaryEmbedding, x, cos, sin);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data   = {1, 2, 3, 4, 5, 6};
    std::vector<float> cos_data = {1, 1, 1};
    std::vector<float> sin_data = {0, 0, 0};
    x.copy_from_host(as_bytes(x_data));
    cos.copy_from_host(as_bytes(cos_data));
    sin.copy_from_host(as_bytes(sin_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data == std::vector<float>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("Op::Reshape same data, different shape (Float32)",
          "[graph][cpu][reshape]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{6}}}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y_data == x_data);
    REQUIRE(y.shape() == shape_t{6});
}

TEST_CASE("Op::Transpose 2x3 -> 3x2 (Float32)", "[graph][cpu][transpose]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0}}}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1, 2, 3,
                                  4, 5, 6};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.shape() == shape_t{3, 2});
    REQUIRE(y_data == std::vector<float>{1, 4,
                                          2, 5,
                                          3, 6});
}

TEST_CASE("Op::Concat 1-D (Float32)", "[graph][cpu][concat]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({2}, DataType::Float32);
    auto b = g.input({3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Concat, ConcatAttrs{0}}, a, b);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data = {1, 2};
    std::vector<float> b_data = {3, 4, 5};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.shape() == shape_t{5});
    REQUIRE(y_data == std::vector<float>{1, 2, 3, 4, 5});
}

TEST_CASE("Op::Concat 2-D along axis=0 (Float32)", "[graph][cpu][concat]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({2, 2}, DataType::Float32);
    auto b = g.input({1, 2}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Concat, ConcatAttrs{0}}, a, b);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data = {1, 2,
                                  3, 4};
    std::vector<float> b_data = {5, 6};
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.shape() == shape_t{3, 2});
    REQUIRE(y_data == std::vector<float>{1, 2,
                                          3, 4,
                                          5, 6});
}

TEST_CASE("Op::Concat 3-D along axis=1 (Float32, KV cache shape)", "[graph][cpu][concat]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    // [H=2, S=2, D=2] concat [H=2, S=1, D=2] -> [H=2, S=3, D=2]
    auto a = g.input({2, 2, 2}, DataType::Float32);
    auto b = g.input({2, 1, 2}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Concat, ConcatAttrs{1}}, a, b);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> a_data = {
        10.0f, 11.0f,
        12.0f, 13.0f,
        20.0f, 21.0f,
        22.0f, 23.0f
    };
    std::vector<float> b_data = {
        14.0f, 15.0f,
        24.0f, 25.0f
    };
    a.copy_from_host(as_bytes(a_data));
    b.copy_from_host(as_bytes(b_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.shape() == shape_t{2, 3, 2});
    std::vector<float> expected = {
        10.0f, 11.0f,
        12.0f, 13.0f,
        14.0f, 15.0f,
        20.0f, 21.0f,
        22.0f, 23.0f,
        24.0f, 25.0f
    };
    REQUIRE(y_data == expected);
}

TEST_CASE("Op::RepeatKV 3-D along axis=0 (Float32, GQA KV repeat)", "[graph][cpu][repeat_kv]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    // [H_kv=2, S=2, D=2] with repeats=3 -> [H_q=6, S=2, D=2]
    auto x = g.input({2, 2, 2}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{3, 0}}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
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

TEST_CASE("Op::RepeatKV identity with repeats=1 (Float32)", "[graph][cpu][repeat_kv]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{1, 0}}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    REQUIRE(y.shape() == shape_t{2, 3});
    REQUIRE(y_data == x_data);
}

TEST_CASE("Slice (Float32, CPU) - axis 0 last row extraction", "[cpu][slice]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4, 3}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {3, 0}, .ends = {4, 3}, .strides = {1, 1}
    }}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    };
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    REQUIRE(y.shape() == shape_t{1, 3});
    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    std::vector<float> expected = {10.f, 11.f, 12.f};
    REQUIRE(y_data == expected);
}

TEST_CASE("Slice (Float32, CPU) - multi-axis with striding and negative indices", "[cpu][slice]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({5, 6}, DataType::Float32);
    // 行索引 [-4, -1) 即 [1, 4) 步长 2 -> 行 1, 3
    // 列索引 [1, 5) 步长 2 -> 列 1, 3
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {-4, 1}, .ends = {-1, 5}, .strides = {2, 2}
    }}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> x_data(30);
    for (std::size_t i = 0; i < 30; ++i) x_data[i] = static_cast<float>(i);
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    REQUIRE(y.shape() == shape_t{2, 2});
    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    // 行 1: 列 1(7), 列 3(9)
    // 行 3: 列 1(19), 列 3(21)
    std::vector<float> expected = {7.f, 9.f, 19.f, 21.f};
    REQUIRE(y_data == expected);
}

TEST_CASE("Slice (Int32, CPU) - token sequence slicing", "[cpu][slice]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({8}, DataType::Int32);
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {2}, .ends = {6}, .strides = {1}
    }}, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    const std::vector<std::int32_t> x_data = {10, 20, 30, 40, 50, 60, 70, 80};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    REQUIRE(y.shape() == shape_t{4});
    std::vector<std::int32_t> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    std::vector<std::int32_t> expected = {30, 40, 50, 60};
    REQUIRE(y_data == expected);
}
