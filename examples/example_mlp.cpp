#include <format>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "velomind.h"

namespace {

    template <typename T>
    std::vector<T> _random_vector(std::size_t n, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<T> out(n);
        for (auto& v : out) v = static_cast<T>(dist(rng));
        return out;
    }

}

int main() {
    using namespace velomind;

    constexpr dim_t BATCH   = 4;
    constexpr dim_t IN_DIM  = 3;
    constexpr dim_t HIDDEN  = 8;
    constexpr dim_t OUT_DIM = 2;

    Graph g;
    auto x  = g.input({BATCH,  IN_DIM},  DataType::Float32);
    auto W1 = g.input({IN_DIM, HIDDEN},  DataType::Float32);
    auto b1 = g.input({BATCH,  HIDDEN},  DataType::Float32);
    auto W2 = g.input({HIDDEN, OUT_DIM}, DataType::Float32);
    auto b2 = g.input({BATCH,  OUT_DIM}, DataType::Float32);

    auto h1 = g.op(Op::Add,  g.op(Op::MatMul, x, W1), b1);
    auto h2 = g.op(Op::Relu, h1);
    auto h3 = g.op(Op::Add,  g.op(Op::MatMul, h2, W2), b2);
    auto y  = g.op(Op::Relu, h3);
    g.mark_output(y);

    auto exec = g.build(DeviceType::CPU);
    if (!exec) {
        std::cerr << "MLP 计算图编译失败\n";
        return 1;
    }
    std::cout << std::format("compiled MLP: {} nodes\n", exec->num_nodes());

    std::mt19937 rng(42);
    auto W1_data = _random_vector<float>(IN_DIM  * HIDDEN, rng);
    auto b1_data = _random_vector<float>(1       * HIDDEN, rng);
    auto W2_data = _random_vector<float>(HIDDEN  * OUT_DIM, rng);
    auto b2_data = _random_vector<float>(1       * OUT_DIM, rng);

    // Add 不支持广播；显式展开共享 bias，避免后续批次越界读取。
    b1_data.resize(BATCH * HIDDEN);
    b2_data.resize(BATCH * OUT_DIM);
    for (dim_t row = 1; row < BATCH; ++row) {
        for (dim_t col = 0; col < HIDDEN; ++col) b1_data[row * HIDDEN + col] = b1_data[col];
        for (dim_t col = 0; col < OUT_DIM; ++col) b2_data[row * OUT_DIM + col] = b2_data[col];
    }

    W1.copy_from_host(std::as_bytes(std::span(W1_data)));
    b1.copy_from_host(std::as_bytes(std::span(b1_data)));
    W2.copy_from_host(std::as_bytes(std::span(W2_data)));
    b2.copy_from_host(std::as_bytes(std::span(b2_data)));

    for (int trial = 0; trial < 3; ++trial) {
        auto x_data = _random_vector<float>(BATCH * IN_DIM, rng);

        x.copy_from_host(std::as_bytes(std::span(x_data)));
        exec->execute();

        std::vector<float> y_data(y.numel());
        y.copy_to_host(std::as_writable_bytes(std::span(y_data)));

        std::cout << std::format("trial {}: y[0..1] = [{:.3f}, {:.3f}]\n",
                                 trial, y_data[0], y_data[1]);
    }

    return 0;
}
