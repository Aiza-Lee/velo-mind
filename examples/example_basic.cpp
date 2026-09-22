#include <cmath>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velomind.h"

int main(int argc, char** argv) {
    using namespace velomind;

    // 解析命令行指定的运行后端，默认使用 CPU
    Device device = Device::cpu();
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            auto d = Device::from_string(argv[++i]);
            if (d) {
                device = *d;
            } else {
                std::cerr << std::format("未知设备类型: {} (可选: cpu, ispc, cuda, vulkan)\n", argv[i]);
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << std::format("用法: {} [--device <cpu|ispc|cuda|vulkan>]\n", argv[0]);
            return 0;
        }
    }

    if (!device.is_available()) {
        std::cerr << std::format("设备 {} 当前不可用，请检查相关驱动或运行环境\n", device.name());
        return 1;
    }

    std::cout << "[VeloMind] 运行基础计算图示例\n";
    std::cout << std::format("目标执行设备: {}\n", device.name());

    // 1. 构建计算图：声明输入并连接节点 (y = (a + b) * c)
    Graph g;
    auto a = g.input({4}, DataType::Float32);
    auto b = g.input({4}, DataType::Float32);
    auto c = g.input({4}, DataType::Float32);

    auto ab = g.op(Op::Add, a, b);
    auto y  = g.op(Op::Mul, ab, c);
    g.mark_output(y);

    // 2. 编译图为可执行对象
    auto exec = g.build(device.type());
    if (!exec) {
        std::cerr << "编译计算图失败\n";
        return 1;
    }
    std::cout << std::format("计算图编译完成，节点总数: {}\n", exec->num_nodes());

    // 3. 填充输入张量数据
    const std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f};
    const std::vector<float> c_data = {2.0f, 2.0f, 2.0f, 2.0f};

    a.copy_from_host(std::as_bytes(std::span(a_data)));
    b.copy_from_host(std::as_bytes(std::span(b_data)));
    c.copy_from_host(std::as_bytes(std::span(c_data)));

    // 4. 执行计算
    exec->execute();

    // 5. 将计算结果回传至 Host
    std::vector<float> y_data(y.numel());
    y.copy_to_host(std::as_writable_bytes(std::span(y_data)));

    std::string out_str;
    for (std::size_t i = 0; i < y_data.size(); ++i) {
        if (i > 0) out_str += ", ";
        out_str += std::format("{:.1f}", y_data[i]);
    }
    std::cout << std::format("计算结果: y = (a + b) * c = [{}]\n", out_str);

    // 校验结果正确性
    bool ok = true;
    for (std::size_t i = 0; i < y_data.size(); ++i) {
        float expected = (a_data[i] + b_data[i]) * c_data[i];
        if (std::abs(y_data[i] - expected) > 1e-5f) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "验证通过: 运算结果与预期数学值一致。\n";
        return 0;
    } else {
        std::cerr << "验证失败: 运算结果与预期不符。\n";
        return 1;
    }
}
