# VeloMind

**VeloMind** 是一个轻量级大语言模型（LLM）前向推理引擎原型，支持在 **CPU**、**ISPC**、**NVIDIA CUDA** 与 **Vulkan** 四种计算后端上高效执行计算图编译与类 LLaMA 架构模型的自回归推理。

> **说明**：本项目主要用于探索异构算子调度、计算图优化编译、DSA 内存复用与低延迟端到端自回归推理机制，处于活跃迭代阶段。

---

## 核心特性

- **四类执行后端**：
  - **CPU**：基于 x86_64 AVX2/FMA 内联优化与分块 GEMM 内核。
  - **ISPC**：利用 Intel SPMD 编译器实现细粒度任务多线程调度与 SIMD 并发。
  - **CUDA**：集成 cuBLAS 矩阵乘法、流式在线 Fused Attention 与专用 CUDA 核函数。
  - **Vulkan**：基于 SPIR-V Compute Shader 实现跨平台 GPU 统一内存推理。
- **静态计算图与内存复用**：
  - 声明式构建 DAG，支持死代码消除（DCE）与零拷贝跨步转置视图融合。
  - 基于活跃度区间的 Best-Fit Decreasing 动态存储分配（DSA），实现连续 Arena 物理内存高效打包与复用。
- **LLM 推理基础设施**：
  - 支持 RMSNorm、SwiGLU、RoPE、Grouped-Query Attention (GQA) 与 INT8/INT4 量化。
  - 增量 KV Cache 状态机与流式 TextStreamer 输出。
  - 内置零 Python/外部依赖的轻量 Safetensors 与 BPE 解析器。

---

## 极速体验 (Quickstart)

```bash
# 1. 编译（以 release 预设为例，纯 CPU 环境可选 pure-cpu）
cmake --preset release && cmake --build --preset release -j$(nproc)

# 2. 运行基础算子图示例 (y = (a + b) * c)
./build/release/examples/example_basic

# 3. 运行 LLM 端到端推理（支持 --synthetic 零下载即刻体验）
./build/release/examples/smollm2/smollm2_chat --synthetic --prompt "Hello!"
```

> **提示**：如需下载真实模型权重进行完整多轮对话，请运行 `./scripts/download_model.sh smollm2`（国内环境推荐加 `--hf-mirror`），随后直接执行 `./build/release/examples/smollm2/smollm2_chat`。

---

## 构图示例

```cpp
#include <velomind.h>

velomind::Graph g;
auto a = g.input({4}, velomind::DataType::Float32);
auto b = g.input({4}, velomind::DataType::Float32);
auto c = g.input({4}, velomind::DataType::Float32);

// 声明计算节点并标记终端输出
auto y = g.op(velomind::Op::Mul, g.op(velomind::Op::Add, a, b), c);
g.mark_output(y);

// 静态编译并在指定硬件设备上执行
auto exec = g.build(velomind::DeviceType::CPU);
exec->execute();
```

---

## 文档索引

- **[快速上手与运行指南](docs/GETTING_STARTED.md)**：模型下载、多后端运行参数、REPL 交互指令、合成模式与 5 级资产寻址机制详述。
- **[计算与算子能力清单](docs/CAPABILITIES.md)**：26 个枚举算子的四后端覆盖状态、精度支持、内存布局契约及机器可读版本 [capabilities_matrix.json](docs/capabilities_matrix.json)。

---

## 许可证

本项目遵循 [MIT License](LICENSE) 开源。
