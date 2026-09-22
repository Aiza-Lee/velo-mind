#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <iostream>
#include "velomind/device.h"
#include "velomind/graph.h"
#include "velomind/executable.h"
#include "test_helpers.h"
#include "llama_graph.h"
#include "llama_rope_mask.h"
#include "../examples/smollm2/model.h"
#include "../examples/smollm2/model_loader.h"

using namespace velomind;
using namespace velomind_test;

TEST_CASE("Executable rejects metadata changes before any kernel runs", "[validation]") {
    Executable unbuilt;
    REQUIRE_THROWS_AS(unbuilt.execute(), std::logic_error);
    Graph g;
    auto x = g.input({2}, DataType::Float32);
    auto y = g.op(Op::Relu, x);
    auto z = g.op(Op::Relu, y);
    auto exec = g.build(DeviceType::CPU);
    x.copy_from_host(as_bytes(std::vector<float>{3, 4}));
    y.copy_from_host(as_bytes(std::vector<float>{-7, -7}));
    z.storage()->shape = {1, 2};
    REQUIRE_THROWS_AS(exec->execute(), std::invalid_argument);
    std::vector<float> values(2);
    y.copy_to_host(as_writeable_bytes(values));
    REQUIRE(values == std::vector<float>{-7, -7});
    z.storage()->shape = {2};
    z.storage()->dtype = DataType::Int32;
    REQUIRE_THROWS_AS(exec->execute(), std::invalid_argument);
    z.storage()->dtype = DataType::Float32;
    auto* buffer = x.data();
    x.storage()->data = nullptr;
    REQUIRE_THROWS_AS(exec->execute(), std::invalid_argument);
    x.storage()->data = buffer;
    REQUIRE_NOTHROW(exec->execute());
}

TEST_CASE("Executable validates external input buffers and retains their owner", "[validation]") {
    Graph other;
    auto foreign = other.input({2}, DataType::Float32);
    Graph g;
    auto x = g.input({2}, DataType::Float32);
    auto y = g.op(Op::Relu, x);
    Executable unbuilt;
    REQUIRE_THROWS_AS(unbuilt.bind_input(x, {}), std::logic_error);
    auto exec = g.build(DeviceType::CPU);
    auto buffer = std::make_shared<std::vector<float>>(std::initializer_list<float>{-3, 4});
    std::weak_ptr<std::vector<float>> weak = buffer;
    auto owner = std::shared_ptr<void>(buffer, buffer->data());
    ExternalBuffer valid{buffer->data(), buffer->size() * sizeof(float), DeviceType::CPU, owner};
    REQUIRE_THROWS_AS(exec->bind_input(foreign, valid), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(y, valid), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(x, ExternalBuffer{valid.data, 4, DeviceType::CPU, owner}), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(x, ExternalBuffer{valid.data, valid.capacity_bytes, DeviceType::ISPC, owner}), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(x, ExternalBuffer{valid.data, valid.capacity_bytes, DeviceType::CPU, {}}), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(x, ExternalBuffer{nullptr, valid.capacity_bytes, DeviceType::CPU, owner}), std::invalid_argument);
    REQUIRE_THROWS_AS(exec->bind_input(x, ExternalBuffer{valid.data, valid.capacity_bytes, DeviceType::CPU, buffer}), std::invalid_argument);
    REQUIRE_NOTHROW(exec->bind_input(x, valid));
    valid.owner.reset();
    owner.reset();
    buffer.reset();
    REQUIRE_FALSE(weak.expired());
    REQUIRE_NOTHROW(exec->execute());
    std::vector<float> result(2);
    y.copy_to_host(as_writeable_bytes(result));
    REQUIRE(result == std::vector<float>{0, 4});
}

TEST_CASE("Graph and tensor handles reject use after lifecycle changes", "[validation]") {
    Graph g;
    auto x = g.input({2}, DataType::Float32);
    REQUIRE_THROWS_AS(x.copy_from_host(as_bytes(std::vector<float>{1, 2})), std::logic_error);
    auto y = g.op(Op::Relu, x);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::logic_error);
    REQUIRE_THROWS_AS(g.input({1}, DataType::Float32), std::logic_error);
    REQUIRE_THROWS_AS(g.op(Op::Relu, x), std::logic_error);
    REQUIRE_NOTHROW(exec->execute());

    Tensor expired;
    std::unique_ptr<Executable> stale;
    {
        auto graph = std::make_unique<Graph>();
        expired = graph->input({1}, DataType::Float32);
        graph->op(Op::Relu, expired);
        stale = graph->build(DeviceType::CPU);
    }
    REQUIRE_FALSE(expired.has_storage());
    REQUIRE(stale->output(expired) == nullptr);
    REQUIRE_THROWS_AS(stale->execute(), std::logic_error);
    REQUIRE_THROWS_AS(stale->bind_input(expired, {}), std::logic_error);
    REQUIRE(y.has_storage());
}

TEST_CASE("External input owner follows graph lifetime and rebind", "[validation]") {
    std::weak_ptr<std::vector<float>> first_weak;
    std::weak_ptr<std::vector<float>> second_weak;
    {
        Graph graph;
        auto input = graph.input({1}, DataType::Float32);
        auto output = graph.op(Op::Relu, input);
        auto executable = graph.build(DeviceType::CPU);
        auto first = std::make_shared<std::vector<float>>(1, -2.0f);
        first_weak = first;
        executable->bind_input(input, ExternalBuffer{
            first->data(), sizeof(float), DeviceType::CPU,
            std::shared_ptr<void>(first, first->data())});
        first.reset();
        REQUIRE_FALSE(first_weak.expired());

        auto second = std::make_shared<std::vector<float>>(1, 3.0f);
        second_weak = second;
        executable->bind_input(input, ExternalBuffer{
            second->data(), sizeof(float), DeviceType::CPU,
            std::shared_ptr<void>(second, second->data())});
        second.reset();
        REQUIRE(first_weak.expired());
        REQUIRE_FALSE(second_weak.expired());
        executable->execute();
        std::vector<float> result(1);
        output.copy_to_host(as_writeable_bytes(result));
        REQUIRE(result[0] == 3.0f);
    }
    REQUIRE(second_weak.expired());
}

TEST_CASE("Graph build allocation failure leaves graph retryable", "[validation]") {
    Graph graph;
    auto input = graph.input({2}, DataType::Float32);
    auto output = graph.op(Op::Relu, input);
    auto original_input = input.shared_storage();
    auto original_output = output.shared_storage();
    struct ScopedFailure {
        std::string previous;
        bool had_previous;
        ScopedFailure() {
            const char* value = std::getenv("VELOMIND_FAIL_GRAPH_BUILD_AFTER_ALLOCATIONS");
            had_previous = value != nullptr;
            if (had_previous) previous = value;
            setenv("VELOMIND_FAIL_GRAPH_BUILD_AFTER_ALLOCATIONS", "1", 1);
        }
        ~ScopedFailure() {
            if (had_previous) setenv("VELOMIND_FAIL_GRAPH_BUILD_AFTER_ALLOCATIONS", previous.c_str(), 1);
            else unsetenv("VELOMIND_FAIL_GRAPH_BUILD_AFTER_ALLOCATIONS");
        }
    };
    {
        ScopedFailure failure;
        REQUIRE_THROWS_AS(graph.build(DeviceType::CPU), std::runtime_error);
    }
    REQUIRE(input.shared_storage() == original_input);
    REQUIRE(output.shared_storage() == original_output);
    REQUIRE(input.data() == nullptr);
    REQUIRE(output.data() == nullptr);
    auto executable = graph.build(DeviceType::CPU);
    input.copy_from_host(as_bytes(std::vector<float>{-1, 2}));
    REQUIRE_NOTHROW(executable->execute());
    std::vector<float> result(2);
    output.copy_to_host(as_writeable_bytes(result));
    REQUIRE(result == std::vector<float>{0, 2});
}

TEST_CASE("Graph build rejects implicit migration of populated storage", "[validation]") {
    auto storage = TensorStorage::allocate(sizeof(float), DeviceType::CPU);
    storage->shape = {1};
    storage->dtype = DataType::Float32;
    Graph graph;
    auto input = graph.input(storage);
    REQUIRE_THROWS_AS(graph.build(DeviceType::ISPC), std::invalid_argument);
    REQUIRE(input.shared_storage() == storage);
    REQUIRE_NOTHROW(graph.build(DeviceType::CPU));
}

static void check_embedding_indices(DeviceType device) {
    if (!Device(device).is_available()) SKIP("设备不可用");
    Graph g;
    auto table = g.input({3, 2}, DataType::Float32);
    auto indices = g.input({2}, DataType::Int32);
    auto generated = device == DeviceType::VULKAN
        ? indices : g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{2}}}, indices);
    auto y = g.op(Op::Embedding, table, generated);
    auto exec = g.build(device);
    table.copy_from_host(as_bytes(std::vector<float>{1, 2, 3, 4, 5, 6}));
    for (auto invalid : {-1, 3, std::numeric_limits<int>::max()}) {
        indices.copy_from_host(as_bytes(std::vector<std::int32_t>{0, invalid}));
        y.copy_from_host(as_bytes(std::vector<float>{-7, -7, -7, -7}));
        REQUIRE_THROWS_AS(exec->execute(), std::invalid_argument);
        std::vector<float> values(4);
        y.copy_to_host(as_writeable_bytes(values));
        REQUIRE(values == std::vector<float>{-7, -7, -7, -7});
    }
    indices.copy_from_host(as_bytes(std::vector<std::int32_t>{2, 0}));
    REQUIRE_NOTHROW(exec->execute());
    std::vector<float> values(4);
    y.copy_to_host(as_writeable_bytes(values));
    REQUIRE(values == std::vector<float>{5, 6, 1, 2});
}

TEST_CASE("Embedding index validation CPU", "[validation][cpu]") { check_embedding_indices(DeviceType::CPU); }
#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("Embedding index validation ISPC", "[validation][ispc]") { check_embedding_indices(DeviceType::ISPC); }
#endif
#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("Embedding index validation CUDA", "[validation][cuda]") { check_embedding_indices(DeviceType::CUDA); }
#endif
#ifdef VELOMIND_ENABLE_VULKAN
TEST_CASE("Embedding index validation Vulkan", "[validation][vulkan]") { check_embedding_indices(DeviceType::VULKAN); }
#endif

TEST_CASE("Graph validates operator shape and attribute contracts", "[validation]") {
    Graph g;
    auto a = g.input({2, 4}, DataType::Float32);
    auto b = g.input({1, 4}, DataType::Float32);
    auto w = g.input({4}, DataType::Float32);
    auto i = g.input({2, 4}, DataType::Int32);
    for (auto op : {Op::Add, Op::Sub, Op::Mul, Op::Div}) {
        REQUIRE_THROWS_AS(g.op(op, a, b), std::invalid_argument);
        REQUIRE_THROWS_AS(g.op(op, a, i), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Relu, ConcatAttrs{}}, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::MatMul, MatMulAttrs{true, false}}, a, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{7}}}, a), std::invalid_argument);
    for (const auto& perm : {std::vector<int>{0}, std::vector<int>{0, 0}, std::vector<int>{-1, 0}, std::vector<int>{0, 2}})
        REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Transpose, TransposeAttrs{perm}}, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Concat, ConcatAttrs{2}}, a, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Concat, ConcatAttrs{1}}, a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Concat, a, w), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{1e-5f, 0}}, a, w), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{-1, -1}}, a, w), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::RMSNorm, a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Embedding, w, i), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Embedding, a, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::RotaryEmbedding, a, w, w), std::invalid_argument);
    auto odd = g.input({2, 3}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::RotaryEmbedding, odd, b, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{0, 0}}, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{-1, 0}}, a), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{2, 5}}, a), std::invalid_argument);
    REQUIRE(g.node_tot() == 0);
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{4, 2}}}, a));
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0}}}, a));
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::Concat, ConcatAttrs{-2}}, a, b));
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{2, -1}}, a));
    REQUIRE_NOTHROW(g.op(Op::RMSNorm, a, w));
}

TEST_CASE("Graph rejects incompatible MatMul shapes without adding nodes", "[validation]") {
    Graph g;
    auto vector = g.input({4}, DataType::Float32);
    auto matrix = g.input({2, 4}, DataType::Float32);
    auto wrong_k = g.input({3, 2}, DataType::Float32);
    auto batched = g.input({2, 4, 3}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::MatMul, vector, matrix), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::MatMul, matrix, wrong_k), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::MatMul, batched, wrong_k), std::invalid_argument);
    REQUIRE(g.node_tot() == 0);
    REQUIRE(g.tensor_count() == 4);
}

TEST_CASE("Graph build revalidates all nodes before allocating", "[validation]") {
    Graph g;
    auto a = g.input({2, 4}, DataType::Float32);
    auto b = g.input({2, 4}, DataType::Float32);
    auto y = g.op(Op::Add, a, b);
    b.storage()->shape = {4, 2};
    REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    REQUIRE(a.data() == nullptr);
    REQUIRE(y.data() == nullptr);
    b.storage()->shape = {2, 4};
    y.storage()->shape = {4, 2};
    REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    REQUIRE(a.data() == nullptr);
    y.storage()->shape = {2, 4};
    REQUIRE_THROWS_AS(g.build(static_cast<DeviceType>(255)), std::invalid_argument);
    REQUIRE(a.data() == nullptr);
    Graph unsupported;
    auto half = unsupported.input({2}, DataType::Float16);
    unsupported.op(Op::Relu, half);
    REQUIRE_THROWS_AS(unsupported.build(DeviceType::CPU), std::invalid_argument);
    REQUIRE(half.data() == nullptr);
    REQUIRE_NOTHROW(g.build(DeviceType::CPU));
}

TEST_CASE("Graph validates tensor sizes before allocation", "[validation]") {
    Graph g;
    const auto huge = std::numeric_limits<dim_t>::max();
    REQUIRE_THROWS_AS(g.input({-1}, DataType::Float32), std::invalid_argument);
    REQUIRE_THROWS_AS(g.input({0, -1}, DataType::Float32), std::invalid_argument);
    REQUIRE_THROWS_AS(g.input({huge, 3}, DataType::Int8), std::invalid_argument);
    REQUIRE_THROWS_AS(g.input({huge}, DataType::Float32), std::invalid_argument);
    REQUIRE_THROWS_AS(g.input({2}, static_cast<DataType>(255)), std::invalid_argument);
    REQUIRE(g.tensor_count() == 0);
    auto empty = g.input({0, huge, huge}, DataType::Float32);
    REQUIRE(empty.nbytes() == 0);
    auto scalar = g.input({}, DataType::Float32);
    REQUIRE(scalar.numel() == 1);
    auto storage = std::make_shared<TensorStorage>();
    storage->shape = {4};
    storage->size_bytes = 1;
    REQUIRE_THROWS_AS(g.input(storage), std::invalid_argument);
    auto x = g.input({2}, DataType::Float32);
    x.storage()->shape = {huge};
    REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    REQUIRE(scalar.storage()->data == nullptr);
}

TEST_CASE("Graph rejects empty operator dimensions before dispatch", "[validation]") {
    Graph g;
    auto empty = g.input({2, 0, 3}, DataType::Float32);
    auto scalar = g.input({}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::Relu, empty), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Add, empty, empty), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Reshape, scalar), std::invalid_argument);
    REQUIRE(g.node_tot() == 0);
}

TEST_CASE("Graph checks backend index width before allocation", "[validation]") {
    auto storage = std::make_shared<TensorStorage>();
    storage->shape = {static_cast<dim_t>(std::numeric_limits<int>::max()) + 1};
    storage->size_bytes = storage_nbytes(*storage);
    Graph g;
    auto x = g.input(storage);
    auto y = g.op(Op::Relu, x);
    REQUIRE_THROWS_AS(g.build(DeviceType::ISPC), std::invalid_argument);
    REQUIRE(x.data() == nullptr);
    REQUIRE(y.data() == nullptr);
}

TEST_CASE("Graph rejects invalid tensor handles and unknown operators", "[validation]") {
    Graph g, other;
    auto x = g.input({2}, DataType::Float32);
    auto foreign = other.input({2}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::Add, x, foreign), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Relu, Tensor{}), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Relu, Tensor{&g, 999}), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(static_cast<Op>(255), x), std::invalid_argument);
    REQUIRE(g.tensor_count() == 1);
    REQUIRE(g.node_tot() == 0);
    REQUIRE(x.storage()->data == nullptr);
}

TEST_CASE("Graph validates operator arity before signature dispatch", "[validation]") {
    Graph g;
    auto x = g.input({2, 2}, DataType::Float32);
    for (auto op : {Op::Add, Op::Sub, Op::Mul, Op::Div, Op::MatMul,
                    Op::Concat, Op::RMSNorm, Op::Embedding, Op::RotaryEmbedding,
                    Op::Relu, Op::Reshape, Op::Transpose, Op::Softmax, Op::RepeatKV,
                    Op::FusedAttention}) {
        CAPTURE(op);
        REQUIRE_THROWS_AS(g.op(op), std::invalid_argument);
        REQUIRE_THROWS_AS(g.op(op, x, x, x, x), std::invalid_argument);
    }
    REQUIRE(g.node_tot() == 0);
    REQUIRE(x.storage()->data == nullptr);
}

TEST_CASE("Graph topological sort correctly handles duplicate inputs", "[validation][topo]") {
    Graph g;
    auto x = g.input({2}, DataType::Float32);
    auto a = g.op(Op::Relu, x);
    auto b = g.op(Op::Add, a, a);
    auto c = g.op(Op::Mul, b, b);
    auto d = g.op(Op::Sub, c, a);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    x.copy_from_host(as_bytes(std::vector<float>{2.0f, -3.0f}));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> res_b(2), res_c(2), res_d(2);
    b.copy_to_host(as_writeable_bytes(res_b));
    c.copy_to_host(as_writeable_bytes(res_c));
    d.copy_to_host(as_writeable_bytes(res_d));

    REQUIRE(res_b == std::vector<float>{4.0f, 0.0f});
    REQUIRE(res_c == std::vector<float>{16.0f, 0.0f});
    REQUIRE(res_d == std::vector<float>{14.0f, 0.0f});
}

TEST_CASE("Graph topological sort handles branch and join structures", "[validation][topo]") {
    Graph g;
    auto x = g.input({2}, DataType::Float32);
    auto n0 = g.op(Op::Relu, x);
    auto n1 = g.op(Op::Add, n0, n0);
    auto n2 = g.op(Op::Mul, n0, n0);
    auto n3 = g.op(Op::Add, n1, n2);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    x.copy_from_host(as_bytes(std::vector<float>{3.0f, -2.0f}));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> res(2);
    n3.copy_to_host(as_writeable_bytes(res));
    REQUIRE(res == std::vector<float>{15.0f, 0.0f});
}

TEST_CASE("Graph topological sort handles independent subgraphs", "[validation][topo]") {
    Graph g;
    auto x1 = g.input({2}, DataType::Float32);
    auto a1 = g.op(Op::Relu, x1);
    auto b1 = g.op(Op::Add, a1, a1);

    auto x2 = g.input({2}, DataType::Float32);
    auto a2 = g.op(Op::Neg, x2);
    auto b2 = g.op(Op::Neg, a2);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    x1.copy_from_host(as_bytes(std::vector<float>{2.0f, -4.0f}));
    x2.copy_from_host(as_bytes(std::vector<float>{5.0f, -7.0f}));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> res1(2), res2(2);
    b1.copy_to_host(as_writeable_bytes(res1));
    b2.copy_to_host(as_writeable_bytes(res2));
    REQUIRE(res1 == std::vector<float>{4.0f, 0.0f});
    REQUIRE(res2 == std::vector<float>{5.0f, -7.0f});
}

TEST_CASE("Graph topological sort handles large DAGs efficiently", "[validation][topo]") {
    Graph g;
    auto in = g.input({2}, DataType::Float32);
    auto cur = in;
    constexpr int kChainLength = 500;
    for (int i = 0; i < kChainLength; ++i) {
        cur = g.op(Op::Relu, cur);
    }
    REQUIRE(g.node_tot() == kChainLength);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    in.copy_from_host(as_bytes(std::vector<float>{1.0f, -1.0f}));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> res(2);
    cur.copy_to_host(as_writeable_bytes(res));
    REQUIRE(res == std::vector<float>{1.0f, 0.0f});
}

TEST_CASE("Graph topological sort rejects cycles and incomplete graphs", "[validation][topo]") {
    SECTION("direct self-loop cycle") {
        Graph g;
        auto x = g.input({2}, DataType::Float32);
        auto y = g.op(Op::Relu, x);

        auto& nodes = GraphTestAccess::get_nodes(g);
        REQUIRE(nodes.size() == 1);
        GraphTestAccess::set_node_inputs(nodes[0], {y.storage()});

        REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    }

    SECTION("two-node cycle") {
        Graph g;
        auto x = g.input({2}, DataType::Float32);
        auto y = g.op(Op::Relu, x);
        auto z = g.op(Op::Relu, y);

        auto& nodes = GraphTestAccess::get_nodes(g);
        REQUIRE(nodes.size() == 2);
        GraphTestAccess::set_node_inputs(nodes[0], {z.storage()});

        REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    }

    SECTION("cycle with acyclic branch") {
        Graph g;
        auto x = g.input({2}, DataType::Float32);
        auto n0 = g.op(Op::Relu, x);
        auto n1 = g.op(Op::Relu, n0);
        auto n2 = g.op(Op::Relu, n1);

        auto& nodes = GraphTestAccess::get_nodes(g);
        REQUIRE(nodes.size() == 3);
        GraphTestAccess::set_node_inputs(nodes[1], {n2.storage()});

        REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::invalid_argument);
    }
}

TEST_CASE("Graph validate executes without allocating memory", "[validation]") {
    Graph g;
    auto a = g.input({2, 3}, DataType::Float32);
    auto b = g.input({2, 3}, DataType::Float32);
    auto c = g.op(Op::Add, a, b);
    auto d = g.op(Op::Relu, c);

    REQUIRE_FALSE(g.is_built());
    REQUIRE_FALSE(g.bound_device().has_value());

    // 独立验证不分配底层缓冲区
    REQUIRE_NOTHROW(g.validate(DeviceType::CPU));
    REQUIRE(a.data() == nullptr);
    REQUIRE(b.data() == nullptr);
    REQUIRE(c.data() == nullptr);
    REQUIRE(d.data() == nullptr);
    REQUIRE_FALSE(g.is_built());
    REQUIRE_FALSE(g.bound_device().has_value());

    SECTION("rejects mismatched shape without allocating") {
        b.storage()->shape = {3, 2};
        REQUIRE_THROWS_AS(g.validate(DeviceType::CPU), std::invalid_argument);
        REQUIRE(a.data() == nullptr);
    }

    SECTION("rejects unsupported kernel without allocating") {
        Graph unsupported;
        auto half = unsupported.input({2}, DataType::Float16);
        unsupported.op(Op::Relu, half);
        REQUIRE_THROWS_AS(unsupported.validate(DeviceType::CPU), std::invalid_argument);
        REQUIRE(half.data() == nullptr);
    }

    SECTION("rejects foreign device populated storage") {
        auto storage = TensorStorage::allocate(sizeof(float), DeviceType::CPU);
        storage->shape = {1};
        storage->dtype = DataType::Float32;
        Graph foreign_g;
        foreign_g.input(storage);
        REQUIRE_THROWS_AS(foreign_g.validate(DeviceType::ISPC), std::invalid_argument);
    }

    SECTION("rejects cycles during validate") {
        Graph cyc_g;
        auto x = cyc_g.input({2}, DataType::Float32);
        auto y = cyc_g.op(Op::Relu, x);
        auto& nodes = GraphTestAccess::get_nodes(cyc_g);
        GraphTestAccess::set_node_inputs(nodes[0], {y.storage()});
        REQUIRE_THROWS_AS(cyc_g.validate(DeviceType::CPU), std::invalid_argument);
    }
}

TEST_CASE("Graph compilation enforces single-assignment device binding and rejects recompilation", "[validation]") {
    Graph g;
    auto a = g.input({2, 2}, DataType::Float32);
    auto b = g.op(Op::Relu, a);

    REQUIRE_FALSE(g.is_built());
    REQUIRE_FALSE(g.bound_device().has_value());

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);
    REQUIRE(g.is_built());
    REQUIRE(g.bound_device() == std::make_optional(DeviceType::CPU));

    // 图编译后不可重编译或再次验证，保证拓扑与设备绑定不可变
    REQUIRE_THROWS_AS(g.build(DeviceType::CPU), std::logic_error);
    REQUIRE_THROWS_AS(g.build(DeviceType::ISPC), std::logic_error);
    REQUIRE_THROWS_AS(g.validate(DeviceType::CPU), std::logic_error);
    REQUIRE_THROWS_AS(g.input({2, 2}, DataType::Float32), std::logic_error);
    REQUIRE_THROWS_AS(g.op(Op::Relu, a), std::logic_error);
}

TEST_CASE("Graph explicit outputs and dead node pruning", "[validation][memory_plan]") {
    Graph g;
    auto x = g.input({4}, DataType::Float32);

    // 活跃分支：x -> a -> b -> out
    auto a = g.op(Op::Relu, x);
    auto b = g.op(Op::Add, a, a);
    auto out = g.op(Op::Mul, b, b);

    // 无用分支：x -> dead1 -> dead2
    auto dead1 = g.op(Op::Relu, x);
    auto dead2 = g.op(Op::Add, dead1, dead1);

    REQUIRE_FALSE(g.has_explicit_outputs());
    REQUIRE(g.node_tot() == 5);

    // 声明 explicit output
    g.mark_output(out);
    REQUIRE(g.has_explicit_outputs());
    REQUIRE(g.outputs().size() == 1);
    REQUIRE(g.is_output_tensor(out.index()));
    REQUIRE_FALSE(g.is_output_tensor(a.index()));
    REQUIRE_FALSE(g.is_output_tensor(dead2.index()));

    // 重复声明幂等
    g.mark_output(out);
    REQUIRE(g.outputs().size() == 1);

    // 编译后无用节点被裁剪，仅活跃节点加入执行体
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);
    REQUIRE(exec->num_nodes() == 3);

    // 无用分支的输出张量不分配底层物理内存
    REQUIRE(dead1.data() == nullptr);
    REQUIRE(dead2.data() == nullptr);
    std::vector<float> dead_buf(4);
    REQUIRE_THROWS_AS(dead2.copy_to_host(as_writeable_bytes(dead_buf)), std::logic_error);

    // 活跃分支正常运行并产生正确结果
    x.copy_from_host(as_bytes(std::vector<float>{1.0f, -2.0f, 3.0f, 0.5f}));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> result(4);
    out.copy_to_host(as_writeable_bytes(result));
    REQUIRE(result == std::vector<float>{4.0f, 0.0f, 36.0f, 1.0f});

    // 编译后拒绝修改输出集合
    REQUIRE_THROWS_AS(g.mark_output(a), std::logic_error);
    REQUIRE_THROWS_AS(g.set_outputs({a}), std::logic_error);

    // 非法或外部张量句柄拒绝声明为输出
    Graph other;
    auto foreign = other.input({4}, DataType::Float32);
    Graph g2;
    REQUIRE_THROWS_AS(g2.mark_output(foreign), std::invalid_argument);
    REQUIRE_THROWS_AS(g2.mark_output(Tensor{}), std::invalid_argument);
    REQUIRE_THROWS_AS(g2.set_outputs({foreign}), std::invalid_argument);
}

TEST_CASE("Graph memory planning and intermediate tensor reuse", "[validation][memory_plan]") {
    Graph g;
    const std::size_t numel = 1024;
    auto x = g.input({static_cast<dim_t>(numel)}, DataType::Float32);

    // 线性单链操作：每个中间张量与后续非重叠活跃区间的张量复用物理槽位
    auto n1 = g.op(Op::Relu, x);
    auto n2 = g.op(Op::Relu, n1);
    auto n3 = g.op(Op::Relu, n2);
    auto n4 = g.op(Op::Relu, n3);
    auto n5 = g.op(Op::Relu, n4);
    auto out = g.op(Op::Relu, n5);

    g.mark_output(out);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    const auto& stats = exec->memory_plan_stats();
    // 5 个中间张量生命周期互不重叠，通过 arena 至少复用 3 个中间值槽位
    REQUIRE(stats.intermediate_tensor_count == 5);
    REQUIRE(stats.reused_tensor_count >= 3);
    REQUIRE(stats.bytes_saved >= 3 * numel * sizeof(float));
    REQUIRE(stats.allocation_count < 7);
    REQUIRE(stats.peak_bytes < 7 * numel * sizeof(float));

    // Graph 同样暴露一致的内存统计指标
    REQUIRE(g.memory_plan_stats().has_value());
    REQUIRE(g.memory_plan_stats()->peak_bytes == stats.peak_bytes);
    REQUIRE(g.memory_plan_stats()->allocation_count == stats.allocation_count);

    // 验证执行数值结果一致性
    std::vector<float> in_vals(numel);
    for (std::size_t i = 0; i < numel; ++i) {
        in_vals[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
    }
    x.copy_from_host(as_bytes(in_vals));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> out_vals(numel);
    out.copy_to_host(as_writeable_bytes(out_vals));

    for (std::size_t i = 0; i < numel; ++i) {
        float expected = in_vals[i] > 0.0f ? in_vals[i] : 0.0f;
        REQUIRE(out_vals[i] == expected);
    }
}

TEST_CASE("Graph memory planning protects observable tensors and shared weights", "[validation][memory_plan]") {
    // 模拟共享权重
    auto weight_storage = TensorStorage::allocate(4 * sizeof(float), DeviceType::CPU);
    weight_storage->shape = {2, 2};
    weight_storage->dtype = DataType::Float32;
    std::vector<float> weights{1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(weight_storage->data, weights.data(), weights.size() * sizeof(float));
    void* original_weight_ptr = weight_storage->data;

    Graph g;
    auto w = g.input(weight_storage);
    auto x = g.input({2, 2}, DataType::Float32);

    auto h1 = g.op(Op::MatMul, x, w);
    auto h2 = g.op(Op::Relu, h1);
    auto h3 = g.op(Op::Add, h2, h2);
    auto out1 = g.op(Op::Relu, h3);
    auto out2 = g.op(Op::Sub, h3, h2);

    // 显式声明多个外部可观察输出
    g.set_outputs({out1, out2});

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    // 共享权重地址未被移动或篡改
    REQUIRE(w.data() == original_weight_ptr);

    x.copy_from_host(as_bytes(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}));
    REQUIRE_NOTHROW(exec->execute());

    // 权重数据保持完整未被覆写
    std::vector<float> weight_readback(4);
    w.copy_to_host(as_writeable_bytes(weight_readback));
    REQUIRE(weight_readback == weights);

    // 两路输出均获得独立专属内存，互不覆盖且结果正确
    std::vector<float> res1(4), res2(4);
    out1.copy_to_host(as_writeable_bytes(res1));
    out2.copy_to_host(as_writeable_bytes(res2));

    REQUIRE(res1 == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});
    REQUIRE(res2 == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
}

TEST_CASE("Graph memory planning cross-backend numerical consistency", "[validation][memory_plan]") {
    const auto run_pipeline = [](DeviceType device) -> std::vector<float> {
        Graph g;
        auto x = g.input({4}, DataType::Float32);
        auto a = g.op(Op::Relu, x);
        auto b = g.op(Op::Add, a, a);
        auto c = g.op(Op::Mul, b, b);
        auto d = g.op(Op::Relu, c);
        g.mark_output(d);

        auto exec = g.build(device);
        x.copy_from_host(as_bytes(std::vector<float>{-1.5f, 2.0f, 0.5f, -3.0f}));
        exec->execute();

        std::vector<float> result(4);
        d.copy_to_host(as_writeable_bytes(result));
        return result;
    };

    auto cpu_res = run_pipeline(DeviceType::CPU);
    REQUIRE(cpu_res == std::vector<float>{0.0f, 16.0f, 1.0f, 0.0f});

    if (Device::ispc().is_available()) {
        auto ispc_res = run_pipeline(DeviceType::ISPC);
        REQUIRE(ispc_res == cpu_res);
    }
}

TEST_CASE("Graph memory planning reduces peak memory compared to unoptimized allocation", "[validation][memory_plan]") {
    // 构造包含 10 层连续前馈的深度网络
    const std::size_t numel = 256;
    const std::size_t layers = 10;

    auto run_test = [&](bool use_explicit_output) -> MemoryPlanStats {
        Graph g;
        auto cur = g.input({static_cast<dim_t>(numel)}, DataType::Float32);
        for (std::size_t i = 0; i < layers; ++i) {
            cur = g.op(Op::Relu, cur);
        }
        if (use_explicit_output) {
            g.mark_output(cur);
        }
        auto exec = g.build(DeviceType::CPU);
        return exec->memory_plan_stats();
    };

    const auto stats_unopt = run_test(false);
    const auto stats_opt = run_test(true);

    // 未优化图为每个张量独立分配
    REQUIRE(stats_unopt.allocation_count == layers + 1);
    REQUIRE(stats_unopt.peak_bytes == (layers + 1) * numel * sizeof(float));
    REQUIRE(stats_unopt.reused_tensor_count == 0);

    // 内存规划图通过活跃区间复用槽位，显著降低分配次数与峰值字节
    REQUIRE(stats_opt.allocation_count < stats_unopt.allocation_count);
    REQUIRE(stats_opt.peak_bytes < stats_unopt.peak_bytes);
    REQUIRE(stats_opt.reused_tensor_count > 0);
    REQUIRE(stats_opt.bytes_saved > 0);
}

TEST_CASE("TensorStorage default strides and contiguity contract", "[validation][layout]") {
    // 标量或空形状
    REQUIRE(default_strides({}).empty());
    REQUIRE(is_shape_contiguous({}, {}));

    // 1D 连续形状
    REQUIRE(default_strides({5}) == stride_t{1});
    REQUIRE(is_shape_contiguous({5}, {1}));

    // 2D 连续形状
    REQUIRE(default_strides({3, 4}) == stride_t{4, 1});
    REQUIRE(is_shape_contiguous({3, 4}, {4, 1}));
    // 2D 非连续形状（如转置）
    REQUIRE_FALSE(is_shape_contiguous({3, 4}, {1, 3}));

    // 3D 连续形状与转置
    REQUIRE(default_strides({2, 3, 4}) == stride_t{12, 4, 1});
    REQUIRE(is_shape_contiguous({2, 3, 4}, {12, 4, 1}));
    REQUIRE_FALSE(is_shape_contiguous({2, 3, 4}, {12, 1, 3}));

    // 包含长度为 1 的退化维度
    REQUIRE(is_shape_contiguous({1, 5}, {5, 1}));
}

TEST_CASE("Tensor layout inspection API reports valid metadata", "[validation][layout]") {
    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto r = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{6}}}, x);

    // 构建前检查默认步长与连续性
    REQUIRE(x.strides() == stride_t{3, 1});
    REQUIRE(x.is_contiguous());
    REQUIRE_FALSE(x.is_alias());
    REQUIRE(x.offset_bytes() == 0);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    // 构建后：连续 Reshape 输出共享输入存储
    REQUIRE(x.capacity_bytes() >= 6 * sizeof(float));
    REQUIRE(r.capacity_bytes() >= 6 * sizeof(float));
    REQUIRE(r.is_alias());
    REQUIRE(r.is_contiguous());
    REQUIRE(r.strides() == stride_t{1});
    REQUIRE(r.data() == x.data());
}

TEST_CASE("Continuous Reshape shares storage with input (zero-copy alias)", "[validation][layout]") {
    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto r1 = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{6}}}, x);
    auto r2 = g.op(OpDescriptor{Op::Reshape, ReshapeAttrs{{3, 2}}}, r1);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    // 链式 Reshape 全程共享底层物理内存
    REQUIRE(r1.data() == x.data());
    REQUIRE(r2.data() == x.data());
    REQUIRE(r1.is_alias());
    REQUIRE(r2.is_alias());

    // 写入规则：修改输入数据对全部共享别名立即可见
    std::vector<float> input_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    x.copy_from_host(as_bytes(input_vals));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> read_r1(6);
    r1.copy_to_host(as_writeable_bytes(read_r1));
    REQUIRE(read_r1 == input_vals);

    std::vector<float> read_r2(6);
    r2.copy_to_host(as_writeable_bytes(read_r2));
    REQUIRE(read_r2 == input_vals);
}

TEST_CASE("Attention path Transpose view feeds into MatMul without copying", "[validation][layout]") {
    // 模拟注意力分数计算拓扑：q @ k^T
    const dim_t H = 2;
    const dim_t seq = 3;
    const dim_t D = 4;

    Graph g;
    auto q = g.input({H, seq, D}, DataType::Float32);
    auto k = g.input({H, seq, D}, DataType::Float32);

    // 转置作为中间计算节点（仅被 MatMul 消费）
    auto k_T = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{0, 2, 1}}}, k);
    auto scores = g.op(Op::MatMul, q, k_T);
    g.mark_output(scores);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    // 在 CPU 上，k_T 形成零拷贝视图
    REQUIRE(k_T.data() == k.data());
    REQUIRE(k_T.is_alias());
    REQUIRE_FALSE(k_T.is_contiguous());
    REQUIRE(k_T.strides() == stride_t{seq * D, 1, D});

    // 准备确定性输入数据
    std::vector<float> q_data(H * seq * D);
    std::vector<float> k_data(H * seq * D);
    for (std::size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(i + 1) * 0.1f;
        k_data[i] = static_cast<float>(i + 1) * 0.2f;
    }
    q.copy_from_host(as_bytes(q_data));
    k.copy_from_host(as_bytes(k_data));

    REQUIRE_NOTHROW(exec->execute());

    // 验证数学正确性：C(h, i, j) = sum_p Q(h, i, p) * K(h, j, p)
    std::vector<float> scores_data(H * seq * seq);
    scores.copy_to_host(as_writeable_bytes(scores_data));

    for (dim_t h = 0; h < H; ++h) {
        for (dim_t i = 0; i < seq; ++i) {
            for (dim_t j = 0; j < seq; ++j) {
                float expected = 0.0f;
                for (dim_t p = 0; p < D; ++p) {
                    float q_val = q_data[(h * seq + i) * D + p];
                    float k_val = k_data[(h * seq + j) * D + p];
                    expected += q_val * k_val;
                }
                float actual = scores_data[(h * seq + i) * seq + j];
                REQUIRE(std::abs(actual - expected) < 1e-5f);
            }
        }
    }
}

TEST_CASE("Explicit output Transpose retains dedicated contiguous buffer", "[validation][layout]") {
    // 若转置结果被标记为图输出，必须分配独立的连续存储以供外部消费
    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    auto t = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{{1, 0}}}, x);
    g.mark_output(t);

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    REQUIRE(t.data() != x.data());
    REQUIRE_FALSE(t.is_alias());
    REQUIRE(t.is_contiguous());

    std::vector<float> x_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    x.copy_from_host(as_bytes(x_vals));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> t_vals(6);
    t.copy_to_host(as_writeable_bytes(t_vals));
    // [2, 3] -> [3, 2]
    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    REQUIRE(t_vals == expected);
}

TEST_CASE("Graph capacity reservation policy and caller hints", "[validation][capacity]") {
    // 默认构造初始化轻量合理容量，避免无意义的固定 1MB 预分配
    Graph g;
    REQUIRE(g.tensor_capacity() >= DEFAULT_TENSOR_CAPACITY);
    REQUIRE(g.node_capacity() >= DEFAULT_NODE_CAPACITY);
    REQUIRE(g.tensor_count() == 0);
    REQUIRE(g.node_count() == 0);
    REQUIRE(g.node_tot() == 0);

    // 调用方可根据预估规模显式提示初始容量
    Graph g_custom(128, 64);
    REQUIRE(g_custom.tensor_capacity() >= 128);
    REQUIRE(g_custom.node_capacity() >= 64);

    // reserve 扩大容量
    g_custom.reserve(256, 128);
    REQUIRE(g_custom.tensor_capacity() >= 256);
    REQUIRE(g_custom.node_capacity() >= 128);

    // reserve 传入较小数值不会缩容
    auto prev_t_cap = g_custom.tensor_capacity();
    auto prev_n_cap = g_custom.node_capacity();
    g_custom.reserve(10, 10);
    REQUIRE(g_custom.tensor_capacity() == prev_t_cap);
    REQUIRE(g_custom.node_capacity() == prev_n_cap);

    // 历史常量保留但仅作为推荐容量参考，非硬上限
    REQUIRE(MAX_TENSORS_PER_GRAPH == 65536);

    // 构建后禁止调用 reserve
    auto x = g_custom.input({2}, DataType::Float32);
    auto y = g_custom.op(Op::Relu, x);
    auto exec = g_custom.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);
    REQUIRE_THROWS_AS(g_custom.reserve(512), std::logic_error);
}

TEST_CASE("Graph dynamic growth confirms capacity is not a hard limit", "[validation][capacity]") {
    // 即使初始容量极小（例如 4 张量、2 节点），Graph 亦可随构图无缝动态扩容，数值正确
    Graph g(4, 2);
    auto in_tensor = g.input({2}, DataType::Float32);
    auto cur = in_tensor;
    const int chain_len = 100;
    for (int i = 0; i < chain_len; ++i) {
        cur = g.op(Op::Relu, cur);
    }

    REQUIRE(g.tensor_count() == static_cast<std::size_t>(chain_len + 1));
    REQUIRE(g.node_count() == static_cast<std::size_t>(chain_len));
    REQUIRE(g.tensor_capacity() >= g.tensor_count());
    REQUIRE(g.node_capacity() >= g.node_count());

    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec != nullptr);

    std::vector<float> input_val = {-5.0f, 10.0f};
    in_tensor.copy_from_host(as_bytes(input_val));
    REQUIRE_NOTHROW(exec->execute());

    std::vector<float> out_val(2);
    cur.copy_to_host(as_writeable_bytes(out_val));
    REQUIRE(out_val[0] == 0.0f);
    REQUIRE(out_val[1] == 10.0f);
}

TEST_CASE("Graph construction time and memory comparison between small and model graphs", "[validation][capacity][benchmark]") {
    // 小图对比：固定预分配与默认按需分配
    const int small_iters = 1000;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < small_iters; ++i) {
        Graph g_old(MAX_TENSORS_PER_GRAPH, 0);
        auto x = g_old.input({2}, DataType::Float32);
        auto a = g_old.op(Op::Relu, x);
        auto b = g_old.op(Op::Abs, a);
        auto c = g_old.op(Op::Tanh, b);
        auto d = g_old.op(Op::Sigmoid, c);
        auto e = g_old.op(Op::Relu, d);
        (void)e;
    }
    auto t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < small_iters; ++i) {
        Graph g_new(DEFAULT_TENSOR_CAPACITY, DEFAULT_NODE_CAPACITY);
        auto x = g_new.input({2}, DataType::Float32);
        auto a = g_new.op(Op::Relu, x);
        auto b = g_new.op(Op::Abs, a);
        auto c = g_new.op(Op::Tanh, b);
        auto d = g_new.op(Op::Sigmoid, c);
        auto e = g_new.op(Op::Relu, d);
        (void)e;
    }
    auto t2 = std::chrono::steady_clock::now();

    auto dur_old_small_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto dur_new_small_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // 内存占用对比：固定 65536 占用 1024KB 指针空间，新默认 64 仅占用 1KB，减少 1024 倍
    const std::size_t mem_old_small_bytes = MAX_TENSORS_PER_GRAPH * sizeof(std::shared_ptr<TensorStorage>);
    const std::size_t mem_new_small_bytes = DEFAULT_TENSOR_CAPACITY * sizeof(std::shared_ptr<TensorStorage>)
                                          + DEFAULT_NODE_CAPACITY * sizeof(Node);

    std::cout << "\n[Benchmark] Small Graph (5 ops, 6 tensors, 1000 iters):" << std::endl;
    std::cout << "  - Fixed 65536 reserve: " << dur_old_small_us << " us (" << (dur_old_small_us / 1000.0) << " us/graph), "
              << mem_old_small_bytes << " bytes (" << (mem_old_small_bytes / 1024) << " KB)" << std::endl;
    std::cout << "  - Default 64 reserve:  " << dur_new_small_us << " us (" << (dur_new_small_us / 1000.0) << " us/graph), "
              << mem_new_small_bytes << " bytes (" << (mem_new_small_bytes / 1024.0) << " KB)" << std::endl;
    std::cout << "  - Memory reduction: " << (static_cast<double>(mem_old_small_bytes) / mem_new_small_bytes) << "x" << std::endl;

    // 模型图对比（SmolLM2 30-layer 全量前向图）：固定 65536 vs 针对性预留
    const auto cfg = velomind::examples::smollm2::kSmolLM2_135M;
    const std::size_t seq = 5;
    const int model_iters = 50;

    auto tm0 = std::chrono::steady_clock::now();
    std::size_t model_tensor_count = 0;
    std::size_t model_node_count = 0;
    for (int i = 0; i < model_iters; ++i) {
        Graph g(MAX_TENSORS_PER_GRAPH, 0);
        auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
        auto weights = velomind::examples::smollm2::bind_smollm2_weights(g, cfg, seq);
        auto logits = velomind::examples::llama::build_forward_graph_prefill(g, cfg, tokens, weights);
        (void)logits;
        if (i == 0) {
            model_tensor_count = g.tensor_count();
            model_node_count = g.node_count();
        }
    }
    auto tm1 = std::chrono::steady_clock::now();

    for (int i = 0; i < model_iters; ++i) {
        Graph g; // 构图内部自动按需调用 g.reserve 进行预估预留
        auto tokens = g.input({static_cast<dim_t>(seq)}, DataType::Int32);
        auto weights = velomind::examples::smollm2::bind_smollm2_weights(g, cfg, seq);
        auto logits = velomind::examples::llama::build_forward_graph_prefill(g, cfg, tokens, weights);
        (void)logits;
    }
    auto tm2 = std::chrono::steady_clock::now();

    auto dur_model_old_us = std::chrono::duration_cast<std::chrono::microseconds>(tm1 - tm0).count();
    auto dur_model_new_us = std::chrono::duration_cast<std::chrono::microseconds>(tm2 - tm1).count();

    const std::size_t model_old_mem_bytes = MAX_TENSORS_PER_GRAPH * sizeof(std::shared_ptr<TensorStorage>);
    const std::size_t model_tailored_mem_bytes = (cfg.num_layers * 25 + 10 + cfg.num_layers * 10 + 20) * sizeof(std::shared_ptr<TensorStorage>)
                                               + (cfg.num_layers * 25 + 10) * sizeof(Node);

    std::cout << "\n[Benchmark] Model Graph (SmolLM2-135M, 30 layers, " << model_tensor_count << " tensors, " << model_node_count << " nodes, " << model_iters << " iters):" << std::endl;
    std::cout << "  - Fixed 65536 reserve: " << dur_model_old_us << " us (" << (dur_model_old_us / static_cast<double>(model_iters)) << " us/graph), "
              << model_old_mem_bytes << " bytes (" << (model_old_mem_bytes / 1024) << " KB)" << std::endl;
    std::cout << "  - Tailored hint reserve: " << dur_model_new_us << " us (" << (dur_model_new_us / static_cast<double>(model_iters)) << " us/graph), "
              << model_tailored_mem_bytes << " bytes (" << (model_tailored_mem_bytes / 1024.0) << " KB)" << std::endl;
    std::cout << "  - Memory reduction: " << (static_cast<double>(model_old_mem_bytes) / model_tailored_mem_bytes) << "x" << std::endl;

    REQUIRE(model_tensor_count > 0);
    REQUIRE(model_node_count > 0);
}

TEST_CASE("Partial D2H and H2D - copy_to_host and copy_from_host with offset and bounds checking",
          "[tensor][partial_copy]") {
    Graph g;
    auto t = g.input({4, 4}, DataType::Float32);
    auto exec = g.build(DeviceType::CPU);
    REQUIRE(exec);

    std::vector<float> src = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f
    };
    t.copy_from_host(std::as_bytes(std::span(src)));

    // 局部 D2H 读取最后一行 (4 个 float，偏移 12 * sizeof(float))
    std::vector<float> last_row(4, 0.0f);
    t.copy_to_host(std::as_writable_bytes(std::span(last_row)), 12 * sizeof(float));
    std::vector<float> expected_last = {12.0f, 13.0f, 14.0f, 15.0f};
    REQUIRE(last_row == expected_last);

    // 局部 D2H 读取第二行 (4 个 float，偏移 4 * sizeof(float))
    std::vector<float> second_row(4, 0.0f);
    t.copy_to_host(std::as_writable_bytes(std::span(second_row)), 4 * sizeof(float));
    std::vector<float> expected_second = {4.0f, 5.0f, 6.0f, 7.0f};
    REQUIRE(second_row == expected_second);

    // 局部 H2D 写入第三行并局部读回验证
    std::vector<float> new_row = {99.0f, 98.0f, 97.0f, 96.0f};
    t.copy_from_host(std::as_bytes(std::span(new_row)), 8 * sizeof(float));

    std::vector<float> read_back(4, 0.0f);
    t.copy_to_host(std::as_writable_bytes(std::span(read_back)), 8 * sizeof(float));
    REQUIRE(read_back == new_row);

    // 越界保护校验：偏移超过容量
    std::vector<float> dummy(2);
    REQUIRE_THROWS_AS(
        t.copy_to_host(std::as_writable_bytes(std::span(dummy)), 15 * sizeof(float)),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        t.copy_from_host(std::as_bytes(std::span(dummy)), 16 * sizeof(float)),
        std::runtime_error);

    // 零字节局部读取安全
    std::span<std::byte> empty_span{};
    REQUIRE_NOTHROW(t.copy_to_host(empty_span, 0));
    REQUIRE_NOTHROW(t.copy_to_host(empty_span, 16 * sizeof(float)));
}

TEST_CASE("Op::Slice validation and signature verification", "[validation][op][slice]") {
    Graph g;
    auto x = g.input({10, 20}, DataType::Float32);

    // 合法切片：指定首轴区间与步长
    REQUIRE_NOTHROW(g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {2}, .ends = {8}, .strides = {2}
    }}, x));

    // 合法切片：支持负索引与多轴
    auto sliced = g.op(OpDescriptor{Op::Slice, SliceAttrs{
        .begins = {-2, 0}, .ends = {10, 20}, .strides = {1, 1}
    }}, x);
    REQUIRE(sliced.shape() == shape_t{2, 20});

    // 属性秩超限
    REQUIRE_THROWS_AS(
        g.op(OpDescriptor{Op::Slice, SliceAttrs{
            .begins = {0, 0, 0}, .ends = {1, 1, 1}, .strides = {1, 1, 1}
        }}, x),
        std::invalid_argument);

    // begins 与 ends 秩不匹配
    REQUIRE_THROWS_AS(
        g.op(OpDescriptor{Op::Slice, SliceAttrs{
            .begins = {0, 0}, .ends = {1}, .strides = {1, 1}
        }}, x),
        std::invalid_argument);

    // 非正步长被拒绝
    REQUIRE_THROWS_AS(
        g.op(OpDescriptor{Op::Slice, SliceAttrs{
            .begins = {0}, .ends = {5}, .strides = {0}
        }}, x),
        std::invalid_argument);

    // begin >= end 被拒绝
    REQUIRE_THROWS_AS(
        g.op(OpDescriptor{Op::Slice, SliceAttrs{
            .begins = {5}, .ends = {3}, .strides = {1}
        }}, x),
        std::invalid_argument);
}
