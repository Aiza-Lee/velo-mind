# 算子与计算能力清单

本文档列出 VeloMind 的硬件后端支持、数据类型、内存契约与算子支持矩阵。机器可读版本见 [capabilities_matrix.json](capabilities_matrix.json)。

---

## 1. 计算设备支持

执行计划在 `Graph::build(device)` 阶段静态绑定至目标硬件后端，运行时直接按拓扑序执行内核。

| 设备 (`DeviceType`) | 后端实现 | 加速特性 | 内存模型 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **`CPU`** | 原生 C++20 + AVX2/FMA | AVX2 分块微面板 GEMM ($M>4$)、流式累加 ($M \le 4$) | 主机内存直接访问，64 字节对齐 | 全算子支持 |
| **`ISPC`** | Intel SPMD 编译器 | Gang 向量化、2D 分块 MatMul、`TaskGroup` 任务调度 | 主机内存直接访问，向量化对齐 | 支持 |
| **`CUDA`** | NVIDIA CUDA + cuBLAS | cuBLAS GEMM / 批量 GEMM、异步流调度、专用内核 | GPU 设备显存，异步流派发 | 支持 (SM 80+) |
| **`VULKAN`** | Vulkan 1.2+ 计算着色器 | SPIR-V 着色器、线程独立命令池与批量录制 | 统一内存 (`HOST_VISIBLE | HOST_COHERENT`) | 支持 |

---

## 2. 数据类型支持

| 类型 (`DataType`) | 大小 | 权重加载 | 算子计算支持 | 用途说明 |
| :--- | :---: | :---: | :--- | :--- |
| **`Float32`** | 4 B | 支持 | CPU / ISPC / CUDA / Vulkan | 主推理精度 |
| **`Int32`** | 4 B | 支持 | CPU / ISPC / CUDA / Vulkan (部分) | 词表索引、形状与切片参数 |
| **`Int8`** | 1 B | 支持 | CPU / ISPC / CUDA | 掩码与量化权重存储 |
| **`Bool`** | 1 B | 支持 | CPU / ISPC | 条件选择与掩码 |
| **`Float16`** | 2 B | 支持 | CPU / CUDA | 低精度权重复用与融合注意力 |
| **`BFloat16`** | 2 B | 支持 | CPU / CUDA | 低精度权重复用与融合注意力 |

---

## 3. 张量内存布局与视图契约

- **行优先连续布局 (C-Contiguous)**：默认步长由 `default_strides(shape)` 生成；`is_contiguous()` 保证连续性，对单元素维度（大小为 1）自动容忍步长跳变。
- **跨步视图与零拷贝**：`Transpose` 可生成跨步视图，共享底层物理存储并仅重排 `shape` 与 `strides`；CPU/CUDA 的 `MatMul` 直接消费转置视图标志位（`trans_a`/`trans_b`），避免显式内存重排。
- **按需局部拷贝**：`Tensor::copy_to_host` 与 `copy_from_host` 支持字节偏移切片，自回归生成时仅回传末行 Logits（$1 \times V$），降低总线传输开销。

---

## 4. 算子支持矩阵

项目定义了 26 个枚举算子，各后端的支持状态与属性契约如下：

| 算子 (`Op`) | 类别 | 输入数 | 属性结构体 (`OpAttrs`) | CPU | ISPC | CUDA | Vulkan | 计算说明 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Add`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | 残差连接；支持同源输入防踏写 |
| **`Sub`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 逐元素减法 |
| **`Mul`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | 逐元素乘法 / 门控乘法；支持标量广播 |
| **`Div`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 逐元素除法 |
| **`Neg`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 符号取反 |
| **`Abs`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 绝对值计算 |
| **`Relu`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | ReLU 激活 |
| **`Sigmoid`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | Sigmoid 激活 |
| **`Tanh`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 双曲正切激活 |
| **`Softmax`** | 归一化 | 1 | `SoftmaxAttrs { axis=-1 }` | F32 | F32 | F32 | F32 | 数值稳定 Softmax（减最大值） |
| **`MatMul`** | 线性代数 | 2 | `MatMulAttrs { trans_a, trans_b }` | F32 | F32 | F32 | F32 | 支持批次广播；支持直接消费转置跨步视图 |
| **`RMSNorm`** | 归一化 | 2 | `RMSNormAttrs { eps=1e-5, axis=-1 }` | F32 | F32 | F32 | F32 | 均方根归一化与仿射缩放 |
| **`Embedding`**| 线性代数 | 2 | `NoAttrs` (表:F32, 索引:I32) | F32 | F32 | F32 | F32 | 词表查找；支持边界检查 |
| **`RotaryEmbedding`**| 注意力机制 | 3 | `NoAttrs` (输入, Cos, Sin) | F32 | F32 | F32 | F32 | RoPE 旋转位置编码 |
| **`RepeatKV`** | 注意力机制 | 1 | `RepeatKVAttrs { repeats, axis=0 }` | F32/I32/I8/Bool/F16/BF16 | F32/I32/I8/Bool | F32/F16/BF16 | F32 | GQA 分组重复映射 |
| **`FusedAttention`** | 注意力机制 | 3 | `FusedAttentionAttrs { scale=1.0, is_causal=true }` | F32/F16/BF16 | — | F32/F16/BF16 | — | 在线 Softmax 融合注意力；内置 GQA 映射 |
| **`QuantizedMatMul`**| 线性代数/量化 | 3 | `QuantizedMatMulAttrs { quant_type, block_size }` | F32/F16/BF16 (激活) + I8 (权重) | — | F32/F16/BF16 (激活) + I8 (权重) | — | 混合精度量化 GEMM/GEMV，支持通道与分块缩放 |
| **`Concat`** | 形状操作 | 2 | `ConcatAttrs { axis=0 }` | F32/I32/I8/Bool | F32/I32/I8/Bool | F32/I32 | F32/I32/I8/Bool | 沿指定轴拼接；用于自回归追加 KV Cache |
| **`Slice`** | 形状操作 | 1 | `SliceAttrs { begins, ends, strides }` | F32/I32/I8/Bool | F32/I32/I8/Bool | F32/I32 | F32 | 多轴切片；用于截取末行隐状态 |
| **`Reshape`** | 形状操作 | 1 | `ReshapeAttrs { shape }` | 零拷贝 | 零拷贝 | 零拷贝 | 零拷贝 | 连续张量零拷贝视图；非连续自动拷贝 |
| **`Transpose`**| 形状操作 | 1 | `TransposeAttrs { perm }` | 零拷贝/置换 | 向量化置换 | 转置视图/内核 | GPU 着色器 | 支持 1D~8D 任意置换与跨步视图 |
| **`Rsqrt`** | 数学运算 | 1 | `NoAttrs` | F32 | F32 | F32 | — | 倒数平方根 |
| **`ReduceSum`**| 统计规约 | 1 | `ReduceAttrs { axes, keepdim }` | F32 | — | — | — | 沿轴求和规约 (CPU 参考实现) |
| **`ReduceMean`**| 统计规约 | 1 | `ReduceAttrs { axes, keepdim }` | F32 | — | — | — | 沿轴均值规约 (CPU 参考实现) |
| **`Conv2D`** | 空间卷积 | 2 | `Conv2DAttrs { padding, stride, dilation, groups }` | F32 | — | — | — | 2D 空间卷积 (CPU 参考实现) |
| **`Broadcast`**| 广播适配 | 1 | `NoAttrs` | F32 | — | — | — | 形状广播适配 (CPU 参考实现) |

---

## 5. 端到端模型支持

### 5.1 验证模型

| 模型 | 架构类型 | 隐藏层 / 头数 / 层数 | 权重格式 | 验证后端 | 运行特性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-135M** | LLaMA (SwiGLU, RMSNorm, GQA) | $d=576, h_q=9, h_{kv}=3, L=30$ | Safetensors (FP32/FP16) | CPU / ISPC / CUDA / Vulkan | Prefill 与增量 Decode、GQA、末行 Logits 优化，跨后端输出对齐 |
| **TinyLlama-1.1B** | LLaMA (SwiGLU, RMSNorm, GQA) | $d=2048, h_q=32, h_{kv}=4, L=22$ | Safetensors + Tokenizer | CPU / ISPC / CUDA / Vulkan | 增量 KV Cache 生成、流式输出、REPL 交互 |
| **ToyLlama** | 小型自包含资产 (~450 KB) | $d=64, h_q=4, h_{kv}=2, L=2$ | 内置测试资产 | CPU / ISPC / CUDA / Vulkan | 离线与 CI/CD 端到端测试 |

### 5.2 核心执行机制

- **Prefill 与 Decode 解耦**：Prefill 阶段并行计算提示词激活并初始化 KV Cache；Decode 阶段单步输入（$S=1$），通过 `Concat` 增量追加 KV Cache。
- **跨图权重共享**：静态权重通过 `std::shared_ptr<TensorStorage>` 引入，多轮或增量解码子图共享只读物理内存。
- **Arena 内存复用**：基于张量活跃区间（Live Range）进行静态内存规划与重叠复用，显著降低中间激活峰值显存与分配开销。
- **四阶段编译流水线**：
  1. `validate`：无分配校验设备支持、DAG 拓扑与形状契约；
  2. 拓扑排序与生命周期分析；
  3. Arena 物理存储规划与异常安全分配；
  4. 算子内核绑定与执行序列生成。

### 5.3 量化支持

- **量化格式**（见 [`include/velomind/quant.h`](../include/velomind/quant.h)）：
  - **INT8 通道对称量化**：按行独立计算缩放比例，激活保持浮点参与混合 GEMM 计算。
  - **INT4 分块量化**：按指定分块大小（如 32）计算比例因子，每字节打包两个 4-bit 权重。
  - **精度评估**：内置 `compute_quantization_metrics` 计算 SNR、RMSE 与余弦相似度。
- **自动量化构图**：`quantize_forward_weights` 批量将注意力及 MLP 线性投影层转换为量化权重，并在图编译中自动替换为 `QuantizedMatMul`。
- **后端内核**：CPU 端通过 AVX2+FMA 流式向量展开解码；CUDA 端提供 Warp 级快速并行规约 GEMV 与 GEMM 内核。

---

## 6. 构建预设

通过 `CMakePresets.json` 提供标准化配置：

| 预设 | 编译选项 | 启用后端 | 说明 |
| :--- | :--- | :--- | :--- |
| **`pure-cpu`** | `-O3 -DNDEBUG` | CPU | 零外部依赖，纯 CPU 基础构建 |
| **`cpu-only`** | `-O3 -DNDEBUG` | CPU + ISPC + Vulkan | 多核 CPU 与 Vulkan GPU 支持（无 CUDA 依赖） |
| **`release`** | `-O3 -DNDEBUG` | CPU + ISPC + CUDA + Vulkan | 全后端开启的优化发行版本 |
| **`debug`** | `-O0 -g` | 全量开启 | 包含完整调试符号与断言 |
| **`relwithdebinfo`** | `-O2 -g -DNDEBUG` | 全量开启 | 优化并保留调试符号，用于性能分析 |

---

## 7. 外部 CMake 引入示例

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_inference_app LANGUAGES CXX)

find_package(VeloMind CONFIG REQUIRED)

add_executable(my_inference_app main.cpp)
target_link_libraries(my_inference_app PRIVATE velomind::velomind_core)
```

调用示例：

```cpp
#include <velomind.h>
#include <iostream>

int main() {
    velomind::Graph graph;
    auto a = graph.input({2, 3}, velomind::DataType::Float32);
    auto b = graph.input({2, 3}, velomind::DataType::Float32);
    auto c = graph.op(velomind::Op::Add, a, b);
    graph.mark_output(c);

    auto exe = graph.build(velomind::DeviceType::CPU);
    std::cout << "Compiled graph with " << exe->num_nodes() << " nodes.\n";
    exe->execute();
    return 0;
}
```
