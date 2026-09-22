#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ispc_runtime.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "../test_helpers.h"

#ifdef VELOMIND_ENABLE_ISPC

TEST_CASE("Element-wise Add round-trip (Float32, ISPC)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.op(Op::Add, a, b);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");
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

TEST_CASE("ISPC op coverage: Add / Sub / Mul / Div / Neg / Abs / Relu / Sigmoid / Tanh", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    auto build_for_ispc = [](Graph& g) -> std::unique_ptr<Executable> {
        return g.build(DeviceType::ISPC);
    };

    {
        Graph g;
        auto a = g.input({4}, DataType::Float32);
        auto b = g.input({4}, DataType::Float32);
        auto c = g.op(Op::Add, a, b);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> a_data = {1, 2, 3, 4};
        std::vector<float> b_data = {10, 20, 30, 40};
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();
        std::vector<float> c_data(c.numel());
        c.copy_to_host(as_writeable_bytes(c_data));
        REQUIRE(c_data[0] == 11.0f);
        REQUIRE(c_data[3] == 44.0f);
    }

    {
        Graph g;
        auto a = g.input({4}, DataType::Float32);
        auto b = g.input({4}, DataType::Float32);
        auto c = g.op(Op::Sub, a, b);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> a_data = {10, 20, 30, 40};
        std::vector<float> b_data = {1, 2, 3, 4};
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();
        std::vector<float> c_data(c.numel());
        c.copy_to_host(as_writeable_bytes(c_data));
        REQUIRE(c_data[0] == 9.0f);
        REQUIRE(c_data[3] == 36.0f);
    }

    {
        Graph g;
        auto a = g.input({4}, DataType::Float32);
        auto b = g.input({4}, DataType::Float32);
        auto c = g.op(Op::Mul, a, b);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> a_data = {1, 2, 3, 4};
        std::vector<float> b_data = {10, 20, 30, 40};
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();
        std::vector<float> c_data(c.numel());
        c.copy_to_host(as_writeable_bytes(c_data));
        REQUIRE(c_data[0] == 10.0f);
        REQUIRE(c_data[3] == 160.0f);
    }

    {
        Graph g;
        auto a = g.input({4}, DataType::Float32);
        auto b = g.input({4}, DataType::Float32);
        auto c = g.op(Op::Div, a, b);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
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
        auto x = g.input({4}, DataType::Float32);
        auto y = g.op(Op::Neg, x);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> x_data = {-2.5f, 0.0f, 1.5f, 3.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == 2.5f);
        REQUIRE(y_data[2] == -1.5f);
    }

    {
        Graph g;
        auto x = g.input({4}, DataType::Float32);
        auto y = g.op(Op::Abs, x);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> x_data = {-2.5f, -0.5f, 0.5f, 2.5f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == 2.5f);
        REQUIRE(y_data[2] == 0.5f);
    }

    {
        Graph g;
        auto x = g.input({4}, DataType::Float32);
        auto y = g.op(Op::Relu, x);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> x_data = {-2.0f, -0.5f, 0.0f, 3.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == 0.0f);
        REQUIRE(y_data[3] == 3.0f);
    }

    {
        Graph g;
        auto x = g.input({3}, DataType::Float32);
        auto y = g.op(Op::Sigmoid, x);
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
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
        auto exec = build_for_ispc(g);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");
        std::vector<float> x_data = {0.0f, 1.0f, -1.0f};
        x.copy_from_host(as_bytes(x_data));
        exec->execute();
        std::vector<float> y_data(y.numel());
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == Catch::Approx(0.0f));
        REQUIRE(y_data[1] == Catch::Approx(0.7615942f));
        REQUIRE(y_data[2] == Catch::Approx(-0.7615942f));
    }
}

TEST_CASE("ISPC MatMul 2D and batched (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    {
        Graph g;
        auto x = g.input({2, 3}, DataType::Float32);
        auto W = g.input({3, 2}, DataType::Float32);
        auto y = g.op(Op::MatMul, x, W);

        auto exec = g.build(DeviceType::ISPC);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");

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
        REQUIRE(y_data[0] == Catch::Approx(4.0f));
        REQUIRE(y_data[1] == Catch::Approx(5.0f));
        REQUIRE(y_data[2] == Catch::Approx(10.0f));
        REQUIRE(y_data[3] == Catch::Approx(11.0f));
    }

    {
        Graph g;
        auto a = g.input({2, 2, 3}, DataType::Float32);
        auto b = g.input({2, 3, 2}, DataType::Float32);
        auto c = g.op(Op::MatMul, a, b);

        auto exec = g.build(DeviceType::ISPC);
        if (exec == nullptr) SKIP("ISPC backend not available in this build.");

        std::vector<float> a_data = {
            1, 2, 3,  4, 5, 6,
            7, 8, 9,  1, 2, 3
        };
        std::vector<float> b_data = {
            1, 0,  0, 1,  1, 1,
            2, 0,  0, 2,  1, 1
        };
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();

        std::vector<float> c_data(c.numel());
        c.copy_to_host(as_writeable_bytes(c_data));

        REQUIRE(c_data[0] == Catch::Approx(4.0f));
        REQUIRE(c_data[1] == Catch::Approx(5.0f));
        REQUIRE(c_data[2] == Catch::Approx(10.0f));
        REQUIRE(c_data[3] == Catch::Approx(11.0f));

        REQUIRE(c_data[4] == Catch::Approx(23.0f));
        REQUIRE(c_data[5] == Catch::Approx(25.0f));
        REQUIRE(c_data[6] == Catch::Approx(5.0f));
        REQUIRE(c_data[7] == Catch::Approx(7.0f));
    }
}

TEST_CASE("ISPC Op::RMSNorm (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 4}, DataType::Float32);
    auto w = g.input({4},    DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{1e-5f, -1}}, x, w);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> x_data = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f
    };
    std::vector<float> w_data = {1.0f, 2.0f, 0.5f, 1.0f};
    x.copy_from_host(as_bytes(x_data));
    w.copy_from_host(as_bytes(w_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == Catch::Approx(1.0f).margin(1e-3f));
    REQUIRE(y_data[1] == Catch::Approx(2.0f).margin(1e-3f));
    REQUIRE(y_data[2] == Catch::Approx(0.5f).margin(1e-3f));
    REQUIRE(y_data[3] == Catch::Approx(1.0f).margin(1e-3f));

    REQUIRE(y_data[4] == Catch::Approx(1.0f).margin(1e-3f));
    REQUIRE(y_data[5] == Catch::Approx(2.0f).margin(1e-3f));
    REQUIRE(y_data[6] == Catch::Approx(0.5f).margin(1e-3f));
    REQUIRE(y_data[7] == Catch::Approx(1.0f).margin(1e-3f));
}

TEST_CASE("ISPC Op::Softmax (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto y = g.op(Op::Softmax, x);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> x_data = {
        0.0f, 0.0f, 0.0f,
        1000.0f, 1000.0f, 1000.0f
    };
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    for (int i = 0; i < 6; ++i) {
        REQUIRE(y_data[i] == Catch::Approx(1.0f / 3.0f).margin(1e-5f));
    }
}

TEST_CASE("ISPC Op::Embedding (Float32, Int32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto table = g.input({4, 3}, DataType::Float32);
    auto idx   = g.input({3},    DataType::Int32);
    auto y     = g.op(Op::Embedding, table, idx);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> table_data = {
        0.0f, 0.1f, 0.2f,
        1.0f, 1.1f, 1.2f,
        2.0f, 2.1f, 2.2f,
        3.0f, 3.1f, 3.2f
    };
    std::vector<int32_t> idx_data = {3, 1, 0};
    table.copy_from_host(as_bytes(table_data));
    idx.copy_from_host(as_bytes(idx_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == Catch::Approx(3.0f));
    REQUIRE(y_data[1] == Catch::Approx(3.1f));
    REQUIRE(y_data[2] == Catch::Approx(3.2f));

    REQUIRE(y_data[3] == Catch::Approx(1.0f));
    REQUIRE(y_data[4] == Catch::Approx(1.1f));
    REQUIRE(y_data[5] == Catch::Approx(1.2f));

    REQUIRE(y_data[6] == Catch::Approx(0.0f));
    REQUIRE(y_data[7] == Catch::Approx(0.1f));
    REQUIRE(y_data[8] == Catch::Approx(0.2f));
}

TEST_CASE("ISPC Op::RotaryEmbedding (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x   = g.input({2, 4}, DataType::Float32);
    auto cos = g.input({1, 2}, DataType::Float32);
    auto sin = g.input({1, 2}, DataType::Float32);
    auto y   = g.op(Op::RotaryEmbedding, x, cos, sin);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3, 4,  5, 6, 7, 8};
    std::vector<float> cos_data = {1.0f, 1.0f};
    std::vector<float> sin_data = {0.0f, 0.0f};

    x.copy_from_host(as_bytes(x_data));
    cos.copy_from_host(as_bytes(cos_data));
    sin.copy_from_host(as_bytes(sin_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    for (std::size_t i = 0; i < 8; ++i) {
        REQUIRE(y_data[i] == Catch::Approx(x_data[i]));
    }
}

TEST_CASE("ISPC Op::Transpose and Reshape (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0}}}, x);
    auto r = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{6}}}, t);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> x_data = {1, 2, 3,
                                 4, 5, 6};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> r_data(r.numel());
    r.copy_to_host(as_writeable_bytes(r_data));

    REQUIRE(r_data[0] == 1.0f);
    REQUIRE(r_data[1] == 4.0f);
    REQUIRE(r_data[2] == 2.0f);
    REQUIRE(r_data[3] == 5.0f);
    REQUIRE(r_data[4] == 3.0f);
    REQUIRE(r_data[5] == 6.0f);
}

TEST_CASE("ISPC Op::Rsqrt (Float32)", "[graph][ispc]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({4}, DataType::Float32);
    auto y = g.op(Op::Rsqrt, x);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    std::vector<float> x_data = {1.0f, 4.0f, 16.0f, 100.0f};
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));

    REQUIRE(y_data[0] == Catch::Approx(1.0f));
    REQUIRE(y_data[1] == Catch::Approx(0.5f));
    REQUIRE(y_data[2] == Catch::Approx(0.25f));
    REQUIRE(y_data[3] == Catch::Approx(0.1f));
}

TEST_CASE("ISPC Op::RepeatKV (Float32)", "[graph][ispc][repeat_kv]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({2, 2, 2}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{3, 0}}, x);

    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

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

TEST_CASE("Slice (Float32, ISPC) - axis 0 last row extraction", "[ispc][slice]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    Graph g;
    auto x = g.input({5, 4}, DataType::Float32);
    auto y = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {4, 0}, .ends = {5, 4}, .strides = {1, 1}
    }}, x);
    auto exec = g.build(DeviceType::ISPC);
    if (exec == nullptr) SKIP("ISPC backend not available in this build.");

    const std::vector<float> x_data = {
         1.f,  2.f,  3.f,  4.f,
         5.f,  6.f,  7.f,  8.f,
         9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f,
        17.f, 18.f, 19.f, 20.f
    };
    x.copy_from_host(as_bytes(x_data));
    exec->execute();

    REQUIRE(y.shape() == shape_t{1, 4});
    std::vector<float> y_data(y.numel());
    y.copy_to_host(as_writeable_bytes(y_data));
    std::vector<float> expected = {17.f, 18.f, 19.f, 20.f};
    REQUIRE(y_data == expected);
}

TEST_CASE("ISPC runtime - thread budget and dynamic configuration", "[ispc][runtime]") {
    using namespace velomind;
    using namespace velomind::backend::ispc;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    std::size_t initial_threads = get_thread_count();
    REQUIRE(initial_threads >= 1);

    // 设置单线程预算并验证计算结果
    set_thread_budget(1);
    REQUIRE(get_thread_count() == 1);
    {
        Graph g;
        auto a = g.input({8}, DataType::Float32);
        auto b = g.input({8}, DataType::Float32);
        auto c = g.op(Op::Add, a, b);
        auto exec = g.build(DeviceType::ISPC);
        REQUIRE(exec != nullptr);

        std::vector<float> a_data = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> b_data = {10, 20, 30, 40, 50, 60, 70, 80};
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();

        std::vector<float> c_data(8);
        c.copy_to_host(as_writeable_bytes(c_data));
        for (std::size_t i = 0; i < 8; ++i) {
            REQUIRE(c_data[i] == a_data[i] + b_data[i]);
        }
    }

    // 设置多线程预算 (4 线程) 并验证矩阵乘法
    set_thread_budget(4);
    REQUIRE(get_thread_count() == 4);
    {
        Graph g;
        auto x = g.input({2, 3}, DataType::Float32);
        auto w = g.input({3, 2}, DataType::Float32);
        auto y = g.op(Op::MatMul, x, w);
        auto exec = g.build(DeviceType::ISPC);
        REQUIRE(exec != nullptr);

        std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
        std::vector<float> w_data = {1, 0, 0, 1, 1, 1};
        x.copy_from_host(as_bytes(x_data));
        w.copy_from_host(as_bytes(w_data));
        exec->execute();

        std::vector<float> y_data(4);
        y.copy_to_host(as_writeable_bytes(y_data));
        REQUIRE(y_data[0] == Catch::Approx(4.0f));
        REQUIRE(y_data[1] == Catch::Approx(5.0f));
        REQUIRE(y_data[2] == Catch::Approx(10.0f));
        REQUIRE(y_data[3] == Catch::Approx(11.0f));
    }

    // 恢复默认线程预算
    set_thread_budget(0);
    REQUIRE(get_thread_count() == initial_threads);
}

TEST_CASE("ISPC runtime - TaskGroup concurrent notification, destruction, and multi-threaded stress",
          "[ispc][task_system][concurrency]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    constexpr int kNumThreads = 6;
    constexpr int kItersPerThread = 20;

    std::vector<std::future<void>> futures;
    futures.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [t]() {
            for (int iter = 0; iter < kItersPerThread; ++iter) {
                Graph g;
                auto a = g.input({64}, DataType::Float32);
                auto b = g.input({64}, DataType::Float32);
                auto c = g.op(Op::Add, a, b);
                auto d = g.op(Op::Mul, c, a);
                auto exec = g.build(DeviceType::ISPC);
                if (!exec) return;

                std::vector<float> a_data(64, static_cast<float>(t + 1));
                std::vector<float> b_data(64, static_cast<float>(iter + 1));
                a.copy_from_host(as_bytes(a_data));
                b.copy_from_host(as_bytes(b_data));
                exec->execute();

                std::vector<float> d_data(64);
                d.copy_to_host(as_writeable_bytes(d_data));
                float expected = (static_cast<float>(t + 1) + static_cast<float>(iter + 1)) * static_cast<float>(t + 1);
                REQUIRE(d_data[0] == Catch::Approx(expected));
                REQUIRE(d_data[63] == Catch::Approx(expected));
            }
        }));
    }

    for (auto& f : futures) {
        REQUIRE_NOTHROW(f.get());
    }
}

TEST_CASE("ISPC small-op serial thresholds vs parallel scaling", "[ispc][thresholds]") {
    using namespace velomind;
    using velomind_test::as_bytes;
    using velomind_test::as_writeable_bytes;

    auto build_ispc = [](Graph& g) -> std::unique_ptr<Executable> {
        return g.build(DeviceType::ISPC);
    };

    // RMSNorm 串行阈值 (rows <= 2 vs rows > 2)
    for (int64_t rows : {1, 2, 4, 8}) {
        Graph g;
        auto x = g.input({rows, 8}, DataType::Float32);
        auto w = g.input({8}, DataType::Float32);
        auto y = g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{1e-5f, -1}}, x, w);
        auto exec = build_ispc(g);
        REQUIRE(exec != nullptr);

        std::vector<float> x_data(static_cast<std::size_t>(rows * 8), 2.0f);
        std::vector<float> w_data(8, 1.0f);
        x.copy_from_host(as_bytes(x_data));
        w.copy_from_host(as_bytes(w_data));
        exec->execute();

        std::vector<float> y_data(static_cast<std::size_t>(rows * 8));
        y.copy_to_host(as_writeable_bytes(y_data));
        for (std::size_t i = 0; i < y_data.size(); ++i) {
            REQUIRE(y_data[i] == Catch::Approx(1.0f).margin(1e-3f));
        }
    }

    // Softmax 串行阈值 (rows <= 2 vs rows > 2)
    for (int64_t rows : {1, 2, 4}) {
        Graph g;
        auto x = g.input({rows, 4}, DataType::Float32);
        auto y = g.op(Op::Softmax, x);
        auto exec = build_ispc(g);
        REQUIRE(exec != nullptr);

        std::vector<float> x_data(static_cast<std::size_t>(rows * 4), 1.0f);
        x.copy_from_host(as_bytes(x_data));
        exec->execute();

        std::vector<float> y_data(static_cast<std::size_t>(rows * 4));
        y.copy_to_host(as_writeable_bytes(y_data));
        for (std::size_t i = 0; i < y_data.size(); ++i) {
            REQUIRE(y_data[i] == Catch::Approx(0.25f).margin(1e-5f));
        }
    }

    // MatMul 单 tile 串行与多 tile 并行
    for (int64_t m : {1, 4}) {
        Graph g;
        auto a = g.input({m, 32}, DataType::Float32);
        auto b = g.input({32, 64}, DataType::Float32);
        auto c = g.op(Op::MatMul, a, b);
        auto exec = build_ispc(g);
        REQUIRE(exec != nullptr);

        std::vector<float> a_data(m * 32, 1.0f);
        std::vector<float> b_data(32 * 64, 0.5f);
        a.copy_from_host(as_bytes(a_data));
        b.copy_from_host(as_bytes(b_data));
        exec->execute();

        std::vector<float> c_data(m * 64);
        c.copy_to_host(as_writeable_bytes(c_data));
        for (std::size_t i = 0; i < c_data.size(); ++i) {
            REQUIRE(c_data[i] == Catch::Approx(16.0f));
        }
    }
}

#endif
