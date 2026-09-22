#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "velomind.h"

namespace {

    constexpr std::uint32_t MAGIC = 0x564D4C31;
    constexpr std::size_t   IN_DIM  = 28 * 28;
    constexpr std::size_t   HIDDEN  = 128;
    constexpr std::size_t   OUT_DIM = 10;
    constexpr std::size_t   BATCH   = 4;

    std::string _cache_path() {
        const char* home = std::getenv("HOME");
        if (home == nullptr) home = "/tmp";
        return std::string(home) + "/.cache/velomind-examples/mnist_mlp.bin";
    }

    struct Weights {
        std::vector<float> W1, b1, W2, b2;
    };

    Weights _generate_synthetic_mnist_weights(std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        Weights w;
        w.W1.assign(IN_DIM  * HIDDEN, 0.0f);
        w.b1.assign(1         * HIDDEN, 0.0f);
        w.W2.assign(HIDDEN   * OUT_DIM, 0.0f);
        w.b2.assign(1         * OUT_DIM, 0.0f);
        for (auto& v : w.W1) v = dist(rng);
        for (auto& v : w.b1) v = dist(rng);
        for (auto& v : w.W2) v = dist(rng);
        for (auto& v : w.b2) v = dist(rng);
        return w;
    }

    Weights _load_or_generate_weights() {
        const auto path = _cache_path();
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::uint32_t magic = 0;
            in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            if (magic == MAGIC) {
                Weights w;
                auto read_floats = [&](std::vector<float>& v, std::size_t n) {
                    v.resize(n);
                    in.read(reinterpret_cast<char*>(v.data()),
                            static_cast<std::streamsize>(n * sizeof(float)));
                };
                read_floats(w.W1, IN_DIM  * HIDDEN);
                read_floats(w.b1, 1        * HIDDEN);
                read_floats(w.W2, HIDDEN  * OUT_DIM);
                read_floats(w.b2, 1        * OUT_DIM);
                if (in) {
                    std::cout << std::format("loaded weights from {}\n", path);
                    return w;
                }
            }
        }

        Weights w = _generate_synthetic_mnist_weights(0xC0FFEE);

        std::ofstream out(path, std::ios::binary);
        if (out) {
            std::uint32_t magic = MAGIC;
            out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            auto write_floats = [&](const std::vector<float>& v) {
                out.write(reinterpret_cast<const char*>(v.data()),
                        static_cast<std::streamsize>(v.size() * sizeof(float)));
            };
            write_floats(w.W1);
            write_floats(w.b1);
            write_floats(w.W2);
            write_floats(w.b2);
            std::cout << std::format("cached generated weights at {}\n", path);
        }
        return w;
    }

    std::vector<float> _random_batch_images(std::vector<int>* labels,
                                        std::mt19937& rng) {
        std::uniform_int_distribution<int> digit(0, 9);
        std::normal_distribution<float> noise(0.0f, 0.1f);
        std::vector<float> img(BATCH * IN_DIM, 0.0f);
        labels->clear();
        labels->reserve(BATCH);
        for (std::size_t b = 0; b < BATCH; ++b) {
            int label = digit(rng);
            labels->push_back(label);
            std::size_t row = static_cast<std::size_t>(label) * 3;
            for (std::size_t c = 0; c < 28; ++c) {
                img[b * IN_DIM + row * 28 + c] = 1.0f + noise(rng);
            }
        }
        return img;
    }

}

int main() {
    using namespace velomind;

    Graph g;
    auto x  = g.input({static_cast<dim_t>(BATCH),  static_cast<dim_t>(IN_DIM)},  DataType::Float32);
    auto W1 = g.input({static_cast<dim_t>(IN_DIM), static_cast<dim_t>(HIDDEN)},  DataType::Float32);
    auto b1 = g.input({static_cast<dim_t>(BATCH), static_cast<dim_t>(HIDDEN)},  DataType::Float32);
    auto W2 = g.input({static_cast<dim_t>(HIDDEN), static_cast<dim_t>(OUT_DIM)}, DataType::Float32);
    auto b2 = g.input({static_cast<dim_t>(BATCH), static_cast<dim_t>(OUT_DIM)}, DataType::Float32);

    auto h1 = g.op(Op::Add,  g.op(Op::MatMul, x, W1), b1);
    auto h2 = g.op(Op::Relu, h1);
    auto h3 = g.op(Op::Add,  g.op(Op::MatMul, h2, W2), b2);
    auto y  = g.op(Op::Relu, h3);

    auto exec = g.build(DeviceType::CPU);
    if (!exec) {
        std::cerr << "MNIST MLP 计算图编译失败\n";
        return 1;
    }
    std::cout << std::format("compiled MNIST MLP: {} nodes\n", exec->num_nodes());

    auto weights = _load_or_generate_weights();
    // 保留每层共享 bias 的含义，按当前 Add 的同形状契约展开批次。
    weights.b1.resize(BATCH * HIDDEN);
    weights.b2.resize(BATCH * OUT_DIM);
    for (std::size_t row = 1; row < BATCH; ++row) {
        for (std::size_t col = 0; col < HIDDEN; ++col) weights.b1[row * HIDDEN + col] = weights.b1[col];
        for (std::size_t col = 0; col < OUT_DIM; ++col) weights.b2[row * OUT_DIM + col] = weights.b2[col];
    }
    W1.copy_from_host(std::as_bytes(std::span(weights.W1)));
    b1.copy_from_host(std::as_bytes(std::span(weights.b1)));
    W2.copy_from_host(std::as_bytes(std::span(weights.W2)));
    b2.copy_from_host(std::as_bytes(std::span(weights.b2)));

    std::mt19937 rng(7);
    std::vector<int> labels;
    auto images = _random_batch_images(&labels, rng);

    x.copy_from_host(std::as_bytes(std::span(images)));
    exec->execute();

    std::vector<float> y_data(y.numel());
    y.copy_to_host(std::as_writable_bytes(std::span(y_data)));

    for (std::size_t b = 0; b < BATCH; ++b) {
        std::size_t argmax = 0;
        for (std::size_t i = 1; i < OUT_DIM; ++i) {
            if (y_data[b * OUT_DIM + i] > y_data[b * OUT_DIM + argmax]) argmax = i;
        }
        std::cout << std::format("image {}: true={}  pred={}\n",
                                 b, labels[b], argmax);
    }
    return 0;
}
