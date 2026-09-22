# VeloMind 计算与算子能力清单 (Capabilities Matrix)

本文档系统梳理 VeloMind 异构推理引擎的公共计算能力、算子支持范围、数据类型、内存布局契约及真实端到端模型支持现状。机器可读版本见 [capabilities_matrix.json](capabilities_matrix.json)。

---

## 1. 计算设备与运行时支持

VeloMind 采用统一的计算图与动态内核解析架构。执行计划在 `Graph::build(device)` 阶段绑定至对应硬件后端，运行时执行无额外虚函数或分支分发开销。

| 设备类型 (`DeviceType`) | 后端实现 | 向量化 / 加速特性 | 内存与传输模型 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **`CPU`** | 原生 C++20 + x86_64 AVX2/FMA 内联优化 | 标量黄金基准、AVX2 打包微面板 GEMM ($M>4$)、流式连续累加 ($M \le 4$) | 主机内存直接访问、64 字节对齐 | **稳定支持 (100% 算子全覆盖)** |
| **`ISPC`** | Intel SPMD Program Compiler | Gang 向量化、2D 瓦片 MatMul、动态任务系统 (`TaskGroup`、8 线程预算、单 Token 串行直通) | 主机内存直接访问、AVX2 向量化对齐 | **稳定支持** |
| **`CUDA`** | NVIDIA CUDA + cuBLAS | cuBLAS 2D / 批量 GEMM (`cublasGemmStridedBatchedEx`)、异步流调度、图生命周期绑定句柄 | GPU 设备显存、异步活动流派发、边界同步 | **稳定支持 (Ampere / Ada Lovelace)** |
| **`VULKAN`** | Vulkan 1.2+ 计算着色器 | 设备侧 SPIR-V 计算着色器 (`transpose.comp`、`matmul.comp` 等)、线程独占命令池、`ScopedVulkanBatch` 批量录制 | `HOST_VISIBLE \| HOST_COHERENT` 零拷贝映射统一内存 | **稳定支持 (跨厂商 GPU)** |

---

## 2. 数据类型支持范围 (Data Types)

VeloMind 定义了统一的 `DataType` 枚举与大小映射：

| 数据类型 (`DataType`) | 存储字节 | 权重加载 (Safetensors) | 算子执行支持 (Compute) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`Float32`** | 4 字节 | 支持 | **全后端原生支持** (CPU / ISPC / CUDA / Vulkan) | 核心单精度浮点推理 |
| **`Int32`** | 4 字节 | 支持 | CPU, ISPC, CUDA, Vulkan (部分算子) | 用于 `Embedding` 词表索引、`Reshape`、`Slice`、`Concat` |
| **`Int8`** | 1 字节 | 支持 | CPU, ISPC, CUDA | 用于逐元素掩码与量化权重存储 (`QuantizedMatMul`) |
| **`Bool`** | 1 字节 | 支持 | CPU, ISPC | 用于逻辑掩码与条件选择 |
| **`Float16`** | 2 字节 | 支持 | CPU, CUDA | 原生低精度存储、`FusedAttention`、`QuantizedMatMul` 与转换 |
| **`BFloat16`** | 2 字节 | 支持 | CPU, CUDA | 原生低精度存储、`FusedAttention`、`QuantizedMatMul` 与转换 |

---

## 3. 张量内存布局与视图契约 (Tensor Layout Contract)

1. **行优先连续布局 (C-Contiguous):**
   - 默认步长计算由 `default_strides(shape)` 生成：$s[i-1] = s[i] \times \text{shape}[i]$。
   - 连续性由 `is_contiguous()` 保证，针对退化维度（长度为 1）自动容忍跳步。
2. **零拷贝跨步视图 (Strided Views):**
   - `Transpose` 算子支持生成跨步视图：共享底层物理存储指针，仅重排逻辑 `shape` 与 `strides`，由 `external_owner` 锚定生命周期。
   - CPU 与 CUDA 的 `MatMul` 内核支持直接消费转置视图（通过 GEMM 的 `trans_a` / `trans_b` 原生硬件转置参数），消除注意力层冗余显存重排与拷贝。
3. **局部切片与按需回传 (Partial Transfer):**
   - `Tensor::copy_to_host(span, src_offset_bytes)` 与 `Tensor::copy_from_host(span, dst_offset_bytes)` 支持按字节偏移局部读写。
   - 在生成任务中，模型最后一行的 Logits 仅需 $1 \times V$ 局部回传，节省 $S$ 倍总线带宽。

---

## 4. 完整算子支持矩阵 (Operator Matrix)

下表详细列出全部 26 个枚举算子在各个后端上的支持状态、输入/属性契约与计算特性：

| 算子 (`Op`) | 类别 | 输入数 | 属性结构体 (`OpAttrs`) | CPU | ISPC | CUDA | Vulkan | 关键计算说明 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Add`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | 残差连接加法；支持自引用同源输入防踏写 |
| **`Sub`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 逐元素减法 |
| **`Mul`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | SwiGLU 门控乘法；支持标量广播 |
| **`Div`** | 逐元素 | 2 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 逐元素除法 |
| **`Neg`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 符号取反 |
| **`Abs`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 绝对值计算 |
| **`Relu`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | ReLU 激活 |
| **`Sigmoid`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | F32 | SwiGLU 激活中的 Sigmoid 门控分支 |
| **`Tanh`** | 逐元素一元 | 1 | `NoAttrs` | F32/I32/I8/Bool | F32 | F32 | — | 双曲正切激活 |
| **`Softmax`** | 归一化 | 1 | `SoftmaxAttrs { axis=-1 }` | F32 | F32 | F32 | F32 | 数值稳定减最大值；长序列（S=2048, 8.3M 元素）零 NaN/Inf 保证 |
| **`MatMul`** | 线性代数 | 2 | `MatMulAttrs { trans_a, trans_b }` | F32 | F32 | F32 | F32 | 支持 N 维批次广播；CPU 分块微面板/流式解码；CUDA cuBLAS Batched；Vulkan 瓦片循环 |
| **`RMSNorm`** | 归一化 | 2 | `RMSNormAttrs { eps=1e-5, axis=-1 }` | F32 | F32 | F32 | F32 | 均方根归一化与可学习缩放；Vulkan 共享内存树状规约 |
| **`Embedding`**| 线性代数 | 2 | `NoAttrs` (表:F32, 索引:I32) | F32 | F32 | F32 | F32 | 词表查找；Vulkan GPU 索引 Gather；越界保护 |
| **`RotaryEmbedding`**| 注意力机制 | 3 | `NoAttrs` (输入, Cos, Sin) | F32 | F32 | F32 | F32 | RoPE 旋转位置编码；支持前后半段跨步旋转与长上下文外推 |
| **`RepeatKV`** | 注意力机制 | 1 | `RepeatKVAttrs { repeats, axis=0 }` | F32/I32/I8/Bool/F16/BF16 | F32/I32/I8/Bool | F32/F16/BF16 | F32 | 原生 GQA 分组重复映射；消除模型加载期 KV 权重广播与显存浪费 |
| **`FusedAttention`** | 注意力机制 | 3 | `FusedAttentionAttrs { scale=1.0, is_causal=true }` | F32/F16/BF16 | — | F32/F16/BF16 | — | 流式在线 Softmax 融合注意力；消除中间 [H, S, S] 分数与权重物化显存，内置原生 GQA 广播映射 |
| **`QuantizedMatMul`**| 线性代数/量化 | 3 | `QuantizedMatMulAttrs { quant_type, block_size }` | F32/F16/BF16 (激活) + I8 (权重) | — | F32/F16/BF16 (激活) + I8 (权重) | — | 异构混合精度量化 GEMM/GEMV：激活支持 FP32/FP16/BF16，权重 INT8/INT4，尺度通道/分块；CPU AVX2+FMA 流式解码与展开累加；CUDA Warp-level GEMV 树状规约与 2D 网格 GEMM；模型权重内存 4x/8x 缩减 |
| **`Concat`** | 形状操作 | 2 | `ConcatAttrs { axis=0 }` | F32/I32/I8/Bool | F32/I32/I8/Bool | F32/I32 | F32/I32/I8/Bool | 任意轴拼接；用于自回归生成单步增量拼接历史 KV Cache |
| **`Slice`** | 形状操作 | 1 | `SliceAttrs { begins, ends, strides }` | F32/I32/I8/Bool | F32/I32/I8/Bool | F32/I32 | F32 | 多轴切片；用于在 LM Head 前裁剪出末行隐状态（`slice_last_token`） |
| **`Reshape`** | 形状操作 | 1 | `ReshapeAttrs { shape }` | 零拷贝 | 零拷贝 | 零拷贝 | 零拷贝 | 连续张量零拷贝复用底层存储；非连续触发安全拷贝 |
| **`Transpose`**| 形状操作 | 1 | `TransposeAttrs { perm }` | 零拷贝/置换 | 向量化置换 | 转置视图/内核 | GPU 着色器 | 支持通用 1D~8D 任意置换；Vulkan 设备侧 `transpose.comp` 着色器 |
| **`Rsqrt`** | 数学运算 | 1 | `NoAttrs` | F32 | F32 | F32 | — | 倒数平方根 |
| **`ReduceSum`**| 统计规约 | 1 | `ReduceAttrs { axes, keepdim }` | F32 | — | — | — | CPU 参考实现 |
| **`ReduceMean`**| 统计规约 | 1 | `ReduceAttrs { axes, keepdim }` | F32 | — | — | — | CPU 参考实现 |
| **`Conv2D`** | 空间卷积 | 2 | `Conv2DAttrs { padding, stride, dilation, groups }` | F32 | — | — | — | CPU 参考实现 |
| **`Broadcast`**| 广播适配 | 1 | `NoAttrs` | F32 | — | — | — | 辅助广播适配 |

---

## 5. 端到端模型支持与推理特性

VeloMind 具备完整开箱即用的现代自回归大语言模型推理流水线：

### 5.1 模型覆盖清单

| 模型规格 | 架构类型 | 隐藏层 / 头数 / 层数 | 权重文件 | 验证后端 | 特性与吞吐表现 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-135M** | LlamaForCausalLM (SwiGLU, RMSNorm, GQA) | $d=576, h_q=9, h_{kv}=3, L=30$ | `model.safetensors` (Float32 / Float16) | CPU, ISPC, CUDA, Vulkan | 5-token Prefill、增量自回归 Decode、原生 GQA、末行 Logits 优化；跨后端误差 $\le 2 \times 10^{-4}$，Argmax 100% 对齐 |
| **TinyLlama-1.1B** | LlamaForCausalLM (SwiGLU, RMSNorm, GQA) | $d=2048, h_q=32, h_{kv}=4, L=22$ | `model.safetensors` + `tokenizer.model` | CPU, ISPC, CUDA, Vulkan | 交互式 REPL 控制台、流式文本输出、KV Cache 增量生成、CPU 解码吞吐达 15,000+ tok/s |
| **ToyLlama (452 KB)** | LlamaForCausalLM 小型化自包含资产 | $d=64, h_q=4, h_{kv}=2, L=2$ | `tests/testdata/toy_llama/` | CPU, ISPC, CUDA, Vulkan | CI/CD 环境离线端到端测试，无须外部下载大模型资产即可保证推理逻辑全绿 |

### 5.2 推理流水线核心机制

1. **增量自回归 Prefill/Decode 状态机:**
   - 首阶段 Prefill：输入前缀 Token 序列，计算完整注意力并输出各层初始 KV 激活，使用 `slice_last_token` 仅投射生成首个 Token 的 Logits。
   - 自回归 Decode：利用 `Concat` 在时间维度累积历史 KV Cache，每个 Step 仅需前向 $S=1$ 单 Token，彻底消除因果下三角全掩码开销。
2. **跨图权重共享 (Cross-Graph Weight Sharing):**
   - 权重通过 `Graph::input(std::shared_ptr<TensorStorage>)` 引入，所有单步解码子图复用同一份只读底层物理内存，无任何构图重分配与拷贝。
3. **图级内存规划与中间值复用 (Memory Planning & Arena):**
   - 构图期自动计算张量活跃区间（Live Range），采用首次适应（First-Fit）算法规划内存 Arena。
   - SmolLM2 模型图规划峰值内存压缩达 8.7 倍，分配次数降低 90% 以上，严格防范同源算子踩踏。
4. **四阶段事务化构建 (Transaction Compilation):**
   - 阶段 1：`Graph::validate(device) const`：零内存分配、快速检查设备支持、DAG 无环性与形状契约。
   - 阶段 2：计划优化与生命周期分析。
   - 阶段 3：Arena 物理内存分配与异常安全回滚。
   - 阶段 4：算子内核单赋值绑定，生成拓扑执行队列 `Executable`。

### 5.3 模型量化支持与低精度推理

VeloMind 提供从浮点权重到低精度整数的统一动态量化、张量存储转换与高性能执行支持：
1. **量化格式体系 (`include/velomind/quant.h`):**
   - **INT8 对称通道量化 (`QuantType::Int8`):** 按输出通道（行）独立计算比例因子 $S_r = \frac{\max |W_r|}{127}$，权重映射为 $Q_r = \text{round}(W_r / S_r)$，浮点激活直接参与混合 GEMM 计算。
   - **INT4 分块量化 (`QuantType::Int4`):** 按指定分块（如 Block Size = 32）计算局部比例因子，打包为两元素 1 字节（低 4 位与高 4 位），实现高达 8x 权重显存压缩。
   - **保真度监控与评估:** 提供 `compute_quantization_metrics` 计算信噪比（SNR）、均方根误差（RMSE）与余弦相似度（Cosine Similarity），INT8 余弦相似度高达 0.999+。
2. **端到端模型量化图构建 (`examples/common/llama/llama_graph.h`):**
   - 提供 `quantize_forward_weights`，将 LLaMA / SmolLM2 线性层（`wq`, `wk`, `wv`, `wo`, `w_gate`, `w_up`, `w_down`）批量转换为 INT8/INT4 存储，并自动构图为 `Op::QuantizedMatMul`。
   - 非量化算子（如 RoPE、RMSNorm、Embedding）自动复用原始底层物理存储，无需重复显存拷贝。
3. **硬件加速内核与吞吐:**
   - **CPU:** AVX2+FMA 针对单 Token 解码（$M=1$）行流式连续读取与 8 路向量展开列累加，减少访存带宽绑定；
   - **CUDA:** Warp 级别树状快速并行规约量化 GEMV 内核与 2D 网格 GEMM，端到端 Argmax 对齐率 100%。

---

## 6. 构建预设与发布验证矩阵

项目通过 `CMakePresets.json` 提供标准化编译预设，保证不同开发与部署环境的可移植性：

| 预设名称 | 编译器选项 | 加速器依赖 | 适用场景与测试覆盖 |
| :--- | :--- | :--- | :--- |
| **`pure-cpu`** | `-O3 -DNDEBUG` | **无任何硬件加速器** (CUDA=OFF, ISPC=OFF, Vulkan=OFF) | 无显卡云服务器、轻量容器、纯 CPU 生产环境部署；CTest **149/149 通过 (100%)** |
| **`cpu-only`** | `-O3 -DNDEBUG` | ISPC (AVX2), Vulkan Compute (无 CUDA) | 现代多核 x86 工作站、集显与跨厂商 GPU 部署；CTest **189/189 通过 (100%)** |
| **`release`** | `-O3 -DNDEBUG` | CPU, ISPC (AVX2), CUDA (SM 89), Vulkan Compute | 完整加速器异构测试与高性能发行；CTest **216/216 通过 (100%)** |
| **`debug`** | `-O0 -g` | 全量后端开启 | 调试符号完整、断言激活、排查内核边界 |
| **`relwithdebinfo`** | `-O2 -g -DNDEBUG` | 全量后端开启 | 带有调试符号的高性能性能剖析与基准测试 |

---

## 7. 部署与外部引入示例

通过标准 CMake Package 机制引入 VeloMind：

```cmake
cmake_minimum_required(VERSION 3.25)
project(my_inference_app LANGUAGES CXX)

find_package(VeloMind CONFIG REQUIRED)

add_executable(my_inference_app main.cpp)
target_link_libraries(my_inference_app PRIVATE velomind::velomind_core)
```

调用示例：
```cpp
#include <velomind/graph.h>
#include <velomind/executable.h>
#include <iostream>

int main() {
    velomind::Graph graph;
    auto a = graph.input({2, 3}, velomind::DataType::Float32);
    auto b = graph.input({2, 3}, velomind::DataType::Float32);
    auto c = graph.op(velomind::Op::Add, a, b);
    graph.mark_output(c);

    auto exe = graph.build(velomind::DeviceType::CPU);
    std::cout << "Successfully compiled graph with " << exe->num_nodes() << " nodes." << std::endl;
    exe->execute();
    return 0;
}
```
