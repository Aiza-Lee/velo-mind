#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include "velomind/device.h"
#include "velomind/graph.h"
#include "velomind/executable.h"
#include "test_helpers.h"

using namespace velomind;
using namespace velomind_test;

static void check_softmax(DeviceType device) {
    if (!Device(device).is_available()) SKIP("设备不可用");
    for (const shape_t& shape : {shape_t{7}, shape_t{2, 3, 5}, shape_t{2, 1, 17}, shape_t{1, 257}}) {
        const int rank = static_cast<int>(shape.size());
        for (int axis = -rank; axis < rank; ++axis) {
            for (bool extreme : {false, true}) {
                CAPTURE(device, shape, axis, extreme);
                Graph g;
                auto x = g.input(shape, DataType::Float32);
                auto y = g.op(OpDescriptor{Op::Softmax, SoftmaxAttrs{axis}}, x);
                auto exec = g.build(device);
                std::vector<float> input(x.numel());
                for (std::size_t i = 0; i < input.size(); ++i)
                    input[i] = extreme ? -1e35f : static_cast<float>(static_cast<int>((i * 13) % 37) - 18) / 4;
                x.copy_from_host(as_bytes(input));
                exec->execute();
                std::vector<float> result(input.size());
                y.copy_to_host(as_writeable_bytes(result));
                const int normalized = axis < 0 ? axis + rank : axis;
                // 逐元素解码坐标，独立于内核的行索引公式构造参考分组。
                std::vector<std::vector<std::size_t>> coords(input.size(), std::vector<std::size_t>(rank));
                for (std::size_t i = 0; i < input.size(); ++i) {
                    auto index = i;
                    for (int d = rank - 1; d >= 0; --d) {
                        coords[i][d] = index % shape[d];
                        index /= shape[d];
                    }
                }
                for (std::size_t i = 0; i < input.size(); ++i) {
                    std::vector<std::size_t> group;
                    for (std::size_t j = 0; j < input.size(); ++j) {
                        bool same = true;
                        for (int d = 0; d < rank; ++d)
                            if (d != normalized && coords[i][d] != coords[j][d]) same = false;
                        if (same) group.push_back(j);
                    }
                    double maximum = -std::numeric_limits<double>::infinity();
                    for (auto j : group) maximum = std::max(maximum, static_cast<double>(input[j]));
                    double sum = 0;
                    for (auto j : group) sum += std::exp(static_cast<double>(input[j]) - maximum);
                    const double expected = std::exp(static_cast<double>(input[i]) - maximum) / sum;
                    REQUIRE(std::isfinite(result[i]));
                    REQUIRE(result[i] == Catch::Approx(expected).margin(2e-6));
                }
            }
        }
    }
}

TEST_CASE("Softmax axes CPU", "[softmax][cpu]") { check_softmax(DeviceType::CPU); }
#ifdef VELOMIND_ENABLE_ISPC
TEST_CASE("Softmax axes ISPC", "[softmax][ispc]") { check_softmax(DeviceType::ISPC); }
#endif
#ifdef VELOMIND_ENABLE_CUDA
TEST_CASE("Softmax axes CUDA", "[softmax][cuda]") { check_softmax(DeviceType::CUDA); }
#endif
#ifdef VELOMIND_ENABLE_VULKAN
TEST_CASE("Softmax axes Vulkan", "[softmax][vulkan]") { check_softmax(DeviceType::VULKAN); }
#endif

TEST_CASE("Softmax rejects invalid metadata before allocation", "[softmax]") {
    for (const shape_t& shape : {shape_t{}, shape_t{0}, shape_t{2, 0, 3}}) {
        Graph g;
        auto x = g.input(shape, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(Op::Softmax, x), std::invalid_argument);
        REQUIRE(x.storage()->data == nullptr);
    }
    for (int axis : {-3, 2, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
        Graph g;
        auto x = g.input({2, 3}, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Softmax, SoftmaxAttrs{axis}}, x), std::invalid_argument);
    }
    Graph g;
    auto x = g.input({2, 3}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::Softmax), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Softmax, x, x), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Softmax, MatMulAttrs{}}, x), std::invalid_argument);
    auto integer = g.input({2, 3}, DataType::Int32);
    REQUIRE_THROWS_AS(g.op(Op::Softmax, integer), std::invalid_argument);
}
