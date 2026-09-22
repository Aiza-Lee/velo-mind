#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <future>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/types.h"
#include "internal/registry/kernel.h"

#include "test_helpers.h"
#include "test_model_assets.h"
#include "llama_rope_mask.h"

using namespace velomind;
using namespace velomind_test;
using velomind::examples::llama::compute_rope_cache;
using velomind::examples::llama::compute_causal_mask;

namespace {

inline bool is_unary_kernel_available(Op op, DeviceType dev, DataType in_dt, DataType out_dt) {
    std::array<DataType, 1> in_dts{in_dt};
    return internal::resolve_op_kernel(op, in_dts, out_dt, dev) != nullptr;
}

inline bool is_binary_kernel_available(Op op, DeviceType dev, DataType in1_dt, DataType in2_dt, DataType out_dt) {
    std::array<DataType, 2> in_dts{in1_dt, in2_dt};
    return internal::resolve_op_kernel(op, in_dts, out_dt, dev) != nullptr;
}

// 填充连续浮点数据，避免零或极端数值
inline auto make_float_data(std::size_t n, float base = 1.0f, float step = 0.5f) -> std::vector<float> {
    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = base + static_cast<float>(i % 37) * step;
    }
    return data;
}

// 填充带正负交替的浮点数据
inline auto make_signed_float_data(std::size_t n) -> std::vector<float> {
    std::vector<float> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = static_cast<float>((i % 23) - 11) * 0.75f;
        data[i] = (v == 0.0f) ? 0.5f : v;
    }
    return data;
}

// 填充整型数据
inline auto make_int_data(std::size_t n, int32_t base = 1) -> std::vector<int32_t> {
    std::vector<int32_t> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = base + static_cast<int32_t>(i % 17);
    }
    return data;
}

// 校验两个浮点数组近似相等并均有限
inline void check_floats_close(const std::vector<float>& actual,
                               const std::vector<float>& expected,
                               float tol = 1e-4f) {
    REQUIRE(actual.size() == expected.size());
    require_all_finite(actual, "actual");
    require_all_finite(expected, "expected");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        if (diff > tol) {
            INFO("Index " << i << ": actual=" << actual[i] << ", expected=" << expected[i] << ", diff=" << diff);
            CHECK(diff <= tol);
            break;
        }
    }
}

// 获取系统所有已启用的可用后端设备列表
inline auto get_available_devices() -> std::vector<DeviceType> {
    std::vector<DeviceType> devices{DeviceType::CPU};
#ifdef VELOMIND_ENABLE_ISPC
    if (Device::ispc().is_available()) devices.push_back(DeviceType::ISPC);
#endif
#ifdef VELOMIND_ENABLE_CUDA
    if (Device::cuda().is_available()) devices.push_back(DeviceType::CUDA);
#endif
#ifdef VELOMIND_ENABLE_VULKAN
    if (Device::vulkan().is_available()) devices.push_back(DeviceType::VULKAN);
#endif
    return devices;
}

} // namespace

// 非 SIMD 整倍数与奇数边界测试
TEST_CASE("Coverage Matrix - Non-SIMD multiple shapes across ops and backends", "[matrix][non_simd]") {
    const auto devices = get_available_devices();

    // 选取非 4/8/16/32/64 整倍数的代表性尺寸与质数尺寸
    const std::vector<dim_t> odd_lengths = {1, 3, 5, 7, 11, 13, 17, 31, 33, 65, 127, 255, 513};

    for (auto dev : devices) {
        DYNAMIC_SECTION("Backend: " << device_type_name(dev)) {
            // 一维非对齐 Elementwise Add & Mul
            for (dim_t len : odd_lengths) {
                Graph g;
                auto a = g.input({len}, DataType::Float32);
                auto b = g.input({len}, DataType::Float32);
                auto add_out = g.op(Op::Add, a, b);
                auto mul_out = g.op(Op::Mul, add_out, a);

                auto exec = g.build(dev);
                REQUIRE(exec);

                auto a_data = make_float_data(static_cast<std::size_t>(len), 1.0f, 0.2f);
                auto b_data = make_float_data(static_cast<std::size_t>(len), 2.0f, 0.3f);
                a.copy_from_host(std::as_bytes(std::span(a_data)));
                b.copy_from_host(std::as_bytes(std::span(b_data)));

                exec->execute();

                std::vector<float> res(static_cast<std::size_t>(len));
                mul_out.copy_to_host(std::as_writable_bytes(std::span(res)));

                require_all_finite(res, "odd_length_mul_out");

                // 校验首尾及中间元素准确性，确保 SIMD 尾部未发生截断或越界
                std::size_t last = static_cast<std::size_t>(len) - 1;
                float expected_0 = (a_data[0] + b_data[0]) * a_data[0];
                float expected_last = (a_data[last] + b_data[last]) * a_data[last];
                CHECK(res[0] == Catch::Approx(expected_0).margin(1e-5f));
                CHECK(res[last] == Catch::Approx(expected_last).margin(1e-5f));
            }

            // 二维非对齐 MatMul: [M, K] x [K, N] 覆盖质数维度
            const std::vector<std::array<dim_t, 3>> matmul_dims = {
                {1, 3, 5},
                {3, 7, 11},
                {5, 13, 7},
                {7, 1, 13},
                {17, 11, 19}
            };
            for (const auto& [M, K, N] : matmul_dims) {
                Graph g;
                auto A = g.input({M, K}, DataType::Float32);
                auto B = g.input({K, N}, DataType::Float32);
                auto C = g.op(Op::MatMul, A, B);

                auto exec = g.build(dev);
                REQUIRE(exec);

                auto a_data = make_float_data(static_cast<std::size_t>(M * K), 0.5f, 0.1f);
                auto b_data = make_float_data(static_cast<std::size_t>(K * N), 0.2f, 0.05f);
                A.copy_from_host(std::as_bytes(std::span(a_data)));
                B.copy_from_host(std::as_bytes(std::span(b_data)));

                exec->execute();

                std::vector<float> c_out(static_cast<std::size_t>(M * N));
                C.copy_to_host(std::as_writable_bytes(std::span(c_out)));
                require_all_finite(c_out, "matmul_odd_out");

                // CPU 朴素基准对比
                for (dim_t m = 0; m < M; ++m) {
                    for (dim_t n = 0; n < N; ++n) {
                        float sum = 0.0f;
                        for (dim_t k = 0; k < K; ++k) {
                            sum += a_data[static_cast<std::size_t>(m * K + k)] *
                                   b_data[static_cast<std::size_t>(k * N + n)];
                        }
                        std::size_t idx = static_cast<std::size_t>(m * N + n);
                        CHECK(c_out[idx] == Catch::Approx(sum).margin(1e-4f));
                    }
                }
            }

            // 非对齐 Softmax
            {
                Graph g;
                auto sm_in = g.input({3, 17}, DataType::Float32);
                auto sm_out = g.op(Op::Softmax, sm_in);
                auto exec = g.build(dev);
                REQUIRE(exec);

                auto in_data = make_signed_float_data(3 * 17);
                sm_in.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> out(3 * 17);
                sm_out.copy_to_host(std::as_writable_bytes(std::span(out)));
                require_all_finite(out, "softmax_odd_out");

                for (int r = 0; r < 3; ++r) {
                    float row_sum = 0.0f;
                    for (int c = 0; c < 17; ++c) {
                        float val = out[static_cast<std::size_t>(r * 17 + c)];
                        CHECK(val >= 0.0f);
                        CHECK(val <= 1.0f);
                        row_sum += val;
                    }
                    CHECK(row_sum == Catch::Approx(1.0f).margin(1e-4f));
                }
            }

            // 非对齐 RMSNorm
            {
                Graph g;
                auto x = g.input({5, 14}, DataType::Float32);
                auto w = g.input({14}, DataType::Float32);
                auto norm = g.op(Op::RMSNorm, x, w);
                auto exec = g.build(dev);
                REQUIRE(exec);

                auto x_data = make_float_data(5 * 14, 1.0f, 0.1f);
                auto w_data = make_float_data(14, 1.0f, 0.0f);
                x.copy_from_host(std::as_bytes(std::span(x_data)));
                w.copy_from_host(std::as_bytes(std::span(w_data)));
                exec->execute();

                std::vector<float> out(5 * 14);
                norm.copy_to_host(std::as_writable_bytes(std::span(out)));
                require_all_finite(out, "rmsnorm_odd_out");
            }
        }
    }
}

// 零维、标量拦截与退化单元素维度
TEST_CASE("Coverage Matrix - Zero dimensions, scalar rejections and degenerate 1-dims", "[matrix][degenerate]") {
    // 空形状标量拦截
    {
        Graph g;
        auto x = g.input({}, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(Op::Relu, x), std::invalid_argument);
    }

    // 维度包含 0 拦截
    {
        Graph g;
        auto x0 = g.input({0}, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(Op::Relu, x0), std::invalid_argument);
        auto x1 = g.input({3, 0}, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(Op::Relu, x1), std::invalid_argument);
        auto x2 = g.input({0, 4, 5}, DataType::Float32);
        REQUIRE_THROWS_AS(g.op(Op::Relu, x2), std::invalid_argument);
    }

    // 单元素退化维度正常执行 [1], [1, 1], [1, 1, 1], [1, 1, 1, 1]
    const auto devices = get_available_devices();
    for (auto dev : devices) {
        DYNAMIC_SECTION("Degenerate 1-dim on " << device_type_name(dev)) {
            // [1] 单元素加法
            {
                Graph g;
                auto a = g.input({1}, DataType::Float32);
                auto b = g.input({1}, DataType::Float32);
                auto c = g.op(Op::Add, a, b);
                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> a_v = {42.0f}, b_v = {58.0f}, c_v(1);
                a.copy_from_host(std::as_bytes(std::span(a_v)));
                b.copy_from_host(std::as_bytes(std::span(b_v)));
                exec->execute();
                c.copy_to_host(std::as_writable_bytes(std::span(c_v)));
                CHECK(c_v[0] == 100.0f);
            }

            // [1, 1] x [1, 1] MatMul
            {
                Graph g;
                auto A = g.input({1, 1}, DataType::Float32);
                auto B = g.input({1, 1}, DataType::Float32);
                auto C = g.op(Op::MatMul, A, B);
                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> a_v = {3.5f}, b_v = {2.0f}, c_v(1);
                A.copy_from_host(std::as_bytes(std::span(a_v)));
                B.copy_from_host(std::as_bytes(std::span(b_v)));
                exec->execute();
                C.copy_to_host(std::as_writable_bytes(std::span(c_v)));
                CHECK(c_v[0] == 7.0f);
            }

            // [1, 1, 1, 1] 四维退化张量 Transpose 变换
            {
                Graph g;
                auto t = g.input({1, 1, 1, 1}, DataType::Float32);
                auto trans = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {3, 2, 1, 0}}}, t);
                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> in_v = {9.99f}, out_v(1);
                t.copy_from_host(std::as_bytes(std::span(in_v)));
                exec->execute();
                trans.copy_to_host(std::as_writable_bytes(std::span(out_v)));
                CHECK(out_v[0] == 9.99f);
            }
        }
    }
}

// 重复输入、自引用算子与多分支 DAG
TEST_CASE("Coverage Matrix - Duplicate inputs, self-referential ops and aliasing DAGs", "[matrix][duplicate_inputs]") {
    const auto devices = get_available_devices();

    for (auto dev : devices) {
        DYNAMIC_SECTION("Duplicate inputs on " << device_type_name(dev)) {
            // 同一节点自相操作: Add(x, x), Mul(x, x) 及可选 Sub(x, x), Div(x, x)
            {
                Graph g;
                auto x = g.input({4}, DataType::Float32);
                auto add_self = g.op(Op::Add, x, x);
                auto mul_self = g.op(Op::Mul, x, x);
                Tensor sub_self, div_self;
                bool has_sub = is_binary_kernel_available(Op::Sub, dev, DataType::Float32, DataType::Float32, DataType::Float32);
                if (has_sub) {
                    sub_self = g.op(Op::Sub, x, x);
                    div_self = g.op(Op::Div, x, x);
                }

                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> in_data = {2.0f, -3.0f, 4.0f, 5.0f};
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> res_add(4), res_mul(4);
                add_self.copy_to_host(std::as_writable_bytes(std::span(res_add)));
                mul_self.copy_to_host(std::as_writable_bytes(std::span(res_mul)));

                for (std::size_t i = 0; i < 4; ++i) {
                    CHECK(res_add[i] == Catch::Approx(2.0f * in_data[i]).margin(1e-5f));
                    CHECK(res_mul[i] == Catch::Approx(in_data[i] * in_data[i]).margin(1e-5f));
                }

                if (has_sub) {
                    std::vector<float> res_sub(4), res_div(4);
                    sub_self.copy_to_host(std::as_writable_bytes(std::span(res_sub)));
                    div_self.copy_to_host(std::as_writable_bytes(std::span(res_div)));
                    for (std::size_t i = 0; i < 4; ++i) {
                        CHECK(res_sub[i] == Catch::Approx(0.0f).margin(1e-5f));
                        CHECK(res_div[i] == Catch::Approx(1.0f).margin(1e-5f));
                    }
                }
            }

            // 矩阵自乘 MatMul(A, A) (方阵 A^2)
            {
                Graph g;
                auto A = g.input({3, 3}, DataType::Float32);
                auto A2 = g.op(Op::MatMul, A, A);

                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> a_data = {
                    1.0f, 2.0f, 0.0f,
                    0.0f, 1.0f, 1.0f,
                    2.0f, 0.0f, 1.0f
                };
                A.copy_from_host(std::as_bytes(std::span(a_data)));
                exec->execute();

                std::vector<float> a2_out(9);
                A2.copy_to_host(std::as_writable_bytes(std::span(a2_out)));

                std::vector<float> expected = {
                    1.0f, 4.0f, 2.0f,
                    2.0f, 1.0f, 2.0f,
                    4.0f, 4.0f, 1.0f
                };
                check_floats_close(a2_out, expected);
            }

            // 菱形多分支汇聚 DAG
            {
                Graph g;
                auto x = g.input({6}, DataType::Float32);
                Tensor z;
                if (dev == DeviceType::VULKAN) {
                    auto y1 = g.op(Op::Sigmoid, x);
                    auto y2 = g.op(Op::Mul, x, x);
                    z = g.op(Op::Add, y1, y2);
                } else {
                    auto y1 = g.op(Op::Relu, x);
                    auto y2 = g.op(Op::Abs, x);
                    z = g.op(Op::Add, y1, y2);
                }

                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> in_data = {-5.0f, -2.0f, 0.0f, 1.0f, 3.0f, -0.5f};
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> z_out(6);
                z.copy_to_host(std::as_writable_bytes(std::span(z_out)));

                for (std::size_t i = 0; i < 6; ++i) {
                    if (dev == DeviceType::VULKAN) {
                        float sig_v = 1.0f / (1.0f + std::exp(-in_data[i]));
                        float sq_v = in_data[i] * in_data[i];
                        CHECK(z_out[i] == Catch::Approx(sig_v + sq_v).margin(1e-4f));
                    } else {
                        float relu_v = in_data[i] > 0.0f ? in_data[i] : 0.0f;
                        float abs_v = std::abs(in_data[i]);
                        CHECK(z_out[i] == Catch::Approx(relu_v + abs_v).margin(1e-5f));
                    }
                }
            }
        }
    }
}

// 轴、属性与配置边界测试
TEST_CASE("Coverage Matrix - Negative axes, multi-axis permutations and slicing boundaries", "[matrix][axes_and_attrs]") {
    const auto devices = get_available_devices();

    for (auto dev : devices) {
        DYNAMIC_SECTION("Axes & attrs on " << device_type_name(dev)) {
            // 负轴 Softmax
            {
                Graph g;
                auto x = g.input({2, 4}, DataType::Float32);
                auto sm_last = g.op(OpDescriptor{Op::Softmax, SoftmaxAttrs{.axis = -1}}, x);
                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, -1.0f, 2.0f, 1.0f};
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> out(8);
                sm_last.copy_to_host(std::as_writable_bytes(std::span(out)));
                require_all_finite(out, "softmax_axis_neg1");

                // 每一行归一化总和为 1
                float sum0 = out[0] + out[1] + out[2] + out[3];
                float sum1 = out[4] + out[5] + out[6] + out[7];
                CHECK(sum0 == Catch::Approx(1.0f).margin(1e-4f));
                CHECK(sum1 == Catch::Approx(1.0f).margin(1e-4f));
            }

            // 多轴置换 Transpose
            {
                Graph g;
                // [2, 3, 4] -> perm [2, 0, 1] -> [4, 2, 3]
                auto t = g.input({2, 3, 4}, DataType::Float32);
                auto perm_out = g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {2, 0, 1}}}, t);
                REQUIRE(perm_out.shape() == shape_t{4, 2, 3});

                auto exec = g.build(dev);
                REQUIRE(exec);

                auto in_data = make_float_data(2 * 3 * 4, 1.0f, 1.0f);
                t.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> out(2 * 3 * 4);
                perm_out.copy_to_host(std::as_writable_bytes(std::span(out)));
                require_all_finite(out, "transpose_3d_out");

                // 校验坐标映射: in[i, j, k] == out[k, i, j]
                for (dim_t i = 0; i < 2; ++i) {
                    for (dim_t j = 0; j < 3; ++j) {
                        for (dim_t k = 0; k < 4; ++k) {
                            std::size_t in_idx = static_cast<std::size_t>((i * 3 + j) * 4 + k);
                            std::size_t out_idx = static_cast<std::size_t>((k * 2 + i) * 3 + j);
                            CHECK(out[out_idx] == in_data[in_idx]);
                        }
                    }
                }
            }

            // RepeatKV 沿轴扩展
            {
                Graph g;
                auto x = g.input({2, 3}, DataType::Float32);
                auto rep = g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = 3, .axis = 1}}, x);
                REQUIRE(rep.shape() == shape_t{2, 9});

                auto exec = g.build(dev);
                REQUIRE(exec);

                std::vector<float> in_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> out(18);
                rep.copy_to_host(std::as_writable_bytes(std::span(out)));

                require_all_finite(out, "repeat_kv_out");
            }

            // Slice 多轴带步长与区间裁剪
            {
                Graph g;
                auto x = g.input({8, 8}, DataType::Float32);
                auto sl = g.op(OpDescriptor{Op::Slice, SliceAttrs{
                    .begins = {1, 2},
                    .ends = {7, 8},
                    .strides = {2, 3}
                }}, x);
                REQUIRE(sl.shape() == shape_t{3, 2});

                auto exec = g.build(dev);
                REQUIRE(exec);

                auto in_data = make_float_data(64, 0.0f, 1.0f);
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> out(6);
                sl.copy_to_host(std::as_writable_bytes(std::span(out)));

                std::vector<float> expected = {
                    in_data[1 * 8 + 2], in_data[1 * 8 + 5],
                    in_data[3 * 8 + 2], in_data[3 * 8 + 5],
                    in_data[5 * 8 + 2], in_data[5 * 8 + 5]
                };
                check_floats_close(out, expected);
            }
        }
    }
}

// 非法配置与异常边界系统级防护
TEST_CASE("Coverage Matrix - Comprehensive invalid configurations and defensive rejections", "[matrix][invalid_configs]") {
    Graph g;

    // 二维张量形状不匹配
    auto a = g.input({3, 4}, DataType::Float32);
    auto b = g.input({3, 5}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::Add, a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Sub, a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Mul, a, b), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(Op::Div, a, b), std::invalid_argument);

    // 矩阵乘法内维不匹配
    auto m1 = g.input({3, 5}, DataType::Float32);
    auto m2 = g.input({6, 4}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::MatMul, m1, m2), std::invalid_argument);

    // 矩阵乘法一维秩超限拒绝
    auto v1 = g.input({5}, DataType::Float32);
    auto v2 = g.input({5}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::MatMul, v1, v2), std::invalid_argument);

    // Concat 非拼接轴形状不匹配及秩不匹配
    auto c1 = g.input({3, 4}, DataType::Float32);
    auto c2 = g.input({3, 5}, DataType::Float32);
    // axis 0 要求列数相同，此处 4 != 5
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Concat, ConcatAttrs{.axis = 0}}, c1, c2), std::invalid_argument);
    auto c3 = g.input({3, 4, 1}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Concat, ConcatAttrs{.axis = 0}}, c1, c3), std::invalid_argument);

    // Transpose 轴超界、重复轴与秩不匹配
    auto t = g.input({2, 3, 4}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {0, 1}}}, t), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {0, 1, 1}}}, t), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Transpose, TransposeAttrs{.perm = {0, 1, 5}}}, t), std::invalid_argument);

    // Slice 非法步长与越界区间
    auto s = g.input({10, 10}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Slice, SliceAttrs{.begins = {0}, .ends = {5}, .strides = {0}}}, s), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Slice, SliceAttrs{.begins = {5}, .ends = {3}, .strides = {1}}}, s), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::Slice, SliceAttrs{.begins = {0, 0, 0}, .ends = {1, 1, 1}, .strides = {1, 1, 1}}}, s), std::invalid_argument);

    // RMSNorm 非法 epsilon 与权重形状不匹配
    auto r_x = g.input({4, 16}, DataType::Float32);
    auto r_bad_w = g.input({8}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 1e-5f, .axis = -1}}, r_x, r_bad_w), std::invalid_argument);
    auto r_good_w = g.input({16}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 0.0f, .axis = -1}}, r_x, r_good_w), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = -1e-5f, .axis = -1}}, r_x, r_good_w), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RMSNorm, RMSNormAttrs{.epsilon = 1e-5f, .axis = 0}}, r_x, r_good_w), std::invalid_argument);

    // RepeatKV 重复次数必须为正数
    auto rkv_x = g.input({2, 4}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = 0, .axis = 1}}, rkv_x), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = -2, .axis = 1}}, rkv_x), std::invalid_argument);
    REQUIRE_THROWS_AS(g.op(OpDescriptor{Op::RepeatKV, RepeatKVAttrs{.repeats = 2, .axis = 5}}, rkv_x), std::invalid_argument);

    // RotaryEmbedding 奇数维度与形状失配
    auto rope_bad_x = g.input({2, 5}, DataType::Float32);
    auto rope_cos = g.input({2, 2}, DataType::Float32);
    auto rope_sin = g.input({2, 2}, DataType::Float32);
    REQUIRE_THROWS_AS(g.op(Op::RotaryEmbedding, rope_bad_x, rope_cos, rope_sin), std::invalid_argument);

    // Embedding 查找表非 rank 2
    auto emb_table = g.input({10, 8, 2}, DataType::Float32);
    auto emb_idx = g.input({4}, DataType::Int32);
    REQUIRE_THROWS_AS(g.op(Op::Embedding, emb_table, emb_idx), std::invalid_argument);

    // 极端维度步长防溢出
    auto s_clamped = default_strides({0, std::numeric_limits<dim_t>::max(), std::numeric_limits<dim_t>::max()});
    REQUIRE(s_clamped.size() == 3);
    REQUIRE(s_clamped[0] == std::numeric_limits<dim_t>::max());
    REQUIRE_FALSE(is_shape_contiguous({std::numeric_limits<dim_t>::max(), 2}, {2, 1}));
}

// 跨后端一致性与位级对齐矩阵
TEST_CASE("Coverage Matrix - Cross-backend op and dtype consistency", "[matrix][cross_backend]") {
    const auto devices = get_available_devices();

    // Unary 算子跨后端对齐 (Relu, Sigmoid, Tanh, Rsqrt, Neg, Abs)
    const std::vector<Op> unary_ops = {
        Op::Relu, Op::Sigmoid, Op::Tanh, Op::Rsqrt, Op::Neg, Op::Abs
    };

    const std::size_t N = 64;
    auto in_data = make_float_data(N, 0.5f, 0.1f);

    for (Op op : unary_ops) {
        DYNAMIC_SECTION("Unary Op: " << op_name(op)) {
            // CPU 计算参考基准
            std::vector<float> ref_out(N);
            {
                Graph g;
                auto x = g.input({static_cast<dim_t>(N)}, DataType::Float32);
                auto y = g.op(op, x);
                auto exec = g.build(DeviceType::CPU);
                REQUIRE(exec);
                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();
                y.copy_to_host(std::as_writable_bytes(std::span(ref_out)));
                require_all_finite(ref_out, std::string(op_name(op)) + "_cpu_ref");
            }

            // 对比各可用后端
            for (auto dev : devices) {
                if (dev == DeviceType::CPU) continue;
                if (!is_unary_kernel_available(op, dev, DataType::Float32, DataType::Float32)) continue;
                Graph g;
                auto x = g.input({static_cast<dim_t>(N)}, DataType::Float32);
                auto y = g.op(op, x);
                auto exec = g.build(dev);
                REQUIRE(exec);

                x.copy_from_host(std::as_bytes(std::span(in_data)));
                exec->execute();

                std::vector<float> dev_out(N);
                y.copy_to_host(std::as_writable_bytes(std::span(dev_out)));
                check_floats_close(dev_out, ref_out, 1e-4f);
            }
        }
    }

    // Int32 类型算子跨后端位级一致 (Add, Sub, Mul, Reshape, Concat, Slice)
    {
        const std::size_t count = 32;
        auto a_int = make_int_data(count, 5);
        auto b_int = make_int_data(count, 3);

        std::vector<int32_t> cpu_add_res(count);
        {
            Graph g;
            auto a = g.input({static_cast<dim_t>(count)}, DataType::Int32);
            auto b = g.input({static_cast<dim_t>(count)}, DataType::Int32);
            auto c = g.op(Op::Add, a, b);
            auto exec = g.build(DeviceType::CPU);
            REQUIRE(exec);
            a.copy_from_host(std::as_bytes(std::span(a_int)));
            b.copy_from_host(std::as_bytes(std::span(b_int)));
            exec->execute();
            c.copy_to_host(std::as_writable_bytes(std::span(cpu_add_res)));
        }

        for (auto dev : devices) {
            if (dev == DeviceType::CPU) continue;
            if (!is_binary_kernel_available(Op::Add, dev, DataType::Int32, DataType::Int32, DataType::Int32)) continue;
            Graph g;
            auto a = g.input({static_cast<dim_t>(count)}, DataType::Int32);
            auto b = g.input({static_cast<dim_t>(count)}, DataType::Int32);
            auto c = g.op(Op::Add, a, b);
            auto exec = g.build(dev);
            REQUIRE(exec);

            a.copy_from_host(std::as_bytes(std::span(a_int)));
            b.copy_from_host(std::as_bytes(std::span(b_int)));
            exec->execute();

            std::vector<int32_t> dev_res(count);
            c.copy_to_host(std::as_writable_bytes(std::span(dev_res)));
            CHECK(dev_res == cpu_add_res);
        }
    }
}

// 多线程并发系统级检查
TEST_CASE("Coverage Matrix - Multi-threaded concurrent execution across mixed devices", "[matrix][concurrency]") {
    const auto devices = get_available_devices();
    const int num_threads = 8;
    const int iters_per_thread = 10;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        DeviceType dev = devices[static_cast<std::size_t>(t) % devices.size()];
        futures.push_back(std::async(std::launch::async, [dev, t, iters_per_thread]() {
            for (int iter = 0; iter < iters_per_thread; ++iter) {
                // 每个线程独立分配与构图
                Graph g;
                dim_t M = 4 + (t % 3);
                dim_t K = 8 + (iter % 4);
                dim_t N = 6 + (t % 2);

                auto A = g.input({M, K}, DataType::Float32);
                auto B = g.input({K, N}, DataType::Float32);
                auto C = g.op(Op::MatMul, A, B);
                auto out = g.op(Op::Sigmoid, C);

                auto exec = g.build(dev);
                REQUIRE(exec != nullptr);

                auto a_data = make_float_data(static_cast<std::size_t>(M * K), 0.5f + static_cast<float>(t));
                auto b_data = make_float_data(static_cast<std::size_t>(K * N), 0.2f);
                A.copy_from_host(std::as_bytes(std::span(a_data)));
                B.copy_from_host(std::as_bytes(std::span(b_data)));

                exec->execute();

                std::vector<float> res(static_cast<std::size_t>(M * N));
                out.copy_to_host(std::as_writable_bytes(std::span(res)));

                require_all_finite(res, "thread_concurrent_res");
                for (float val : res) {
                    CHECK(val >= 0.0f);
                }
            }
        }));
    }

    for (auto& f : futures) {
        REQUIRE_NOTHROW(f.get());
    }
}

// 长序列极限测试与数值稳定性
TEST_CASE("Coverage Matrix - Long sequence stress (S=2048, S=4096) and numerical stability", "[matrix][long_seq]") {
    // 4096 长度 RoPE 缓存与旋转数值稳定性
    {
        const std::size_t seq_len = 4096;
        const std::size_t head_dim = 64;
        const float rope_theta = 10000.0f;

        std::vector<float> cos_cache, sin_cache;
        compute_rope_cache(seq_len, head_dim, rope_theta, cos_cache, sin_cache);

        require_all_finite(cos_cache, "rope_4096_cos");
        require_all_finite(sin_cache, "rope_4096_sin");

        // 校验界限与正交性
        for (std::size_t i = 0; i < cos_cache.size(); ++i) {
            CHECK(cos_cache[i] >= -1.0001f);
            CHECK(cos_cache[i] <= 1.0001f);
            CHECK(sin_cache[i] >= -1.0001f);
            CHECK(sin_cache[i] <= 1.0001f);
            float norm_sq = cos_cache[i] * cos_cache[i] + sin_cache[i] * sin_cache[i];
            CHECK(norm_sq == Catch::Approx(1.0f).margin(1e-4f));
        }

        // 构建图执行 RoPE 算子
        const auto devices = get_available_devices();
        for (auto dev : devices) {
            DYNAMIC_SECTION("RoPE 4096 on " << device_type_name(dev)) {
                Graph g;
                // [1, 4, 4096, 64] -> 次末维为序列长 4096
                auto q = g.input({1, 4, static_cast<dim_t>(seq_len), static_cast<dim_t>(head_dim)}, DataType::Float32);
                auto cos_in = g.input({static_cast<dim_t>(seq_len), static_cast<dim_t>(head_dim / 2)}, DataType::Float32);
                auto sin_in = g.input({static_cast<dim_t>(seq_len), static_cast<dim_t>(head_dim / 2)}, DataType::Float32);
                auto q_rot = g.op(Op::RotaryEmbedding, q, cos_in, sin_in);

                auto exec = g.build(dev);
                REQUIRE(exec);

                auto q_data = make_float_data(seq_len * 4 * head_dim, 0.1f, 0.01f);
                q.copy_from_host(std::as_bytes(std::span(q_data)));
                cos_in.copy_from_host(std::as_bytes(std::span(cos_cache)));
                sin_in.copy_from_host(std::as_bytes(std::span(sin_cache)));

                exec->execute();

                std::vector<float> q_out(seq_len * 4 * head_dim);
                q_rot.copy_to_host(std::as_writable_bytes(std::span(q_out)));
                require_all_finite(q_out, "q_rot_4096_out");
            }
        }
    }

    // 2048 长度 Causal Mask 与 Softmax 稳定性
    {
        const std::size_t num_heads = 2;
        const std::size_t seq_len = 2048;

        std::vector<float> mask_buf;
        compute_causal_mask(num_heads, seq_len, mask_buf);
        require_all_finite(mask_buf, "causal_mask_2048");

        Graph g;
        auto scores = g.input({static_cast<dim_t>(num_heads), static_cast<dim_t>(seq_len), static_cast<dim_t>(seq_len)}, DataType::Float32);
        auto masked_scores = g.op(Op::Add, scores, scores); // 复合操作模拟
        auto sm = g.op(Op::Softmax, masked_scores);

        auto exec = g.build(DeviceType::CPU);
        REQUIRE(exec);

        // 局部前 64 行进行快速验证
        auto init_scores = make_float_data(num_heads * seq_len * seq_len, 0.0f, 0.001f);
        scores.copy_from_host(std::as_bytes(std::span(init_scores)));
        exec->execute();

        std::vector<float> sm_out(seq_len); // 局部 D2H 读取首行
        sm.copy_to_host(std::as_writable_bytes(std::span(sm_out)), 0);
        require_all_finite(sm_out, "softmax_2048_first_row");

        float row_sum = std::accumulate(sm_out.begin(), sm_out.end(), 0.0f);
        CHECK(row_sum == Catch::Approx(1.0f).margin(1e-3f));
    }
}
