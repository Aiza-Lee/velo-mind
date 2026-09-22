# VeloMind

**VeloMind** 是一个轻量级大语言模型（LLM）前向推理引擎原型。支持在 CPU、ISPC、NVIDIA CUDA 以及 Vulkan 计算后端上运行基础计算图与类 LLaMA 架构模型的推理。

> **说明**：本项目目前主要用于探索异构算子调度、计算图静态编译与自回归推理机制，尚未正式发布。

---

## 核心特性

- **四类执行后端**：
  - **CPU**：基于 x86_64 AVX2/FMA 向量化展开与分块 GEMM 内核。
  - **ISPC**：利用 Intel SPMD 编译器实现任务系统多线程调度与 SIMD 并发。
  - **CUDA**：集成 cuBLAS 矩阵乘法与专用 CUDA 核函数。
  - **Vulkan**：基于 SPIR-V Compute Shader 实现跨平台 GPU 推理。
- **静态计算图与内存复用**：
  - 声明式构建 DAG，支持死代码消除（DCE）。
  - 基于活跃区间的中间张量物理内存规划与复用，显著降低峰值内存占用。
- **LLM 推理基础设施**：
  - 支持 RMSNorm、SwiGLU、RoPE、Grouped-Query Attention 等 LLaMA 组件。
  - 增量 KV Cache 与自回归生成状态机。
  - 内置零外部依赖的轻量 Safetensors 二进制解析器。

---

## 快速开始

可直接编译运行纯代码构建的基础算子图示例（[examples/example_basic.cpp](examples/example_basic.cpp)）：

### 1. 编译示例

根据本地环境选择合适的构建预设（纯 CPU 环境选择 `pure-cpu`，具备工具链可选择 `release`）：

```bash
# 配置构建环境（以 release 预设为例）
cmake --preset release

# 仅编译基础示例程序
cmake --build --preset release --target example_basic -j$(nproc)
```

### 2. 执行与后端验证

```bash
# 默认在 CPU 上执行：y = (a + b) * c
./build/release/examples/example_basic

# 可显式指定其它硬件后端运行（如 ispc / cuda / vulkan）：
./build/release/examples/example_basic --device ispc
./build/release/examples/example_basic --device cuda
./build/release/examples/example_basic --device vulkan
```

示例核心逻辑：

```cpp
#include <velomind.h>

velomind::Graph g;
auto a = g.input({4}, velomind::DataType::Float32);
auto b = g.input({4}, velomind::DataType::Float32);
auto c = g.input({4}, velomind::DataType::Float32);

// 声明计算节点并标记终端输出
auto y = g.op(velomind::Op::Mul, g.op(velomind::Op::Add, a, b), c);
g.mark_output(y);

// 编译并在指定硬件设备上执行
auto exec = g.build(velomind::DeviceType::CPU);
exec->execute();
```

---

## 关于 LLM 端到端示例

仓库在 [examples/](examples/) 下提供了基于 TinyLlama 与 SmolLM2 的完整端到端推理流水线实现：

- **当前状态**：LLM 完整示例（如 `tinyllama_interactive` 和 `smollm2_interactive`）目前硬编码依赖开发者本机目录下的 `.safetensors` 模型权重与分词器文件。


---

## 许可证

本项目遵循 [MIT License](LICENSE) 开源。
