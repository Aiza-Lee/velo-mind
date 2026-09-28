# VeloMind 快速上手与运行指南

本文档提供 VeloMind 的构建配置、基础示例运行、LLM 端到端模型推理以及交互式控制台的完整操作指引。

---

## 1. 构建与编译指南

VeloMind 使用标准 CMake Presets 组织构建配置。根据本地开发环境具备的硬件工具链选择适合的预设：

| 预设名称 (`preset`) | 说明 | 依赖要求 |
| :--- | :--- | :--- |
| **`release`** (推荐) | 全后端优化构建（启用 -O3 -DNDEBUG） | GCC/Clang, ISPC, CUDA Toolkit (可选), Vulkan SDK (可选) |
| **`pure-cpu`** | 严格纯 CPU 构建，彻底禁用 CUDA、ISPC 与 Vulkan | 仅需标准 C++23 编译器 |
| **`cpu-only`** | 仅启用 CPU、ISPC 与 Vulkan，不依赖 NVIDIA CUDA 工具链 | GCC/Clang, ISPC, Vulkan SDK |
| **`debug`** | 调试版本（-O0 -g），包含 CUDA 调试符号 | 调试工具链 |
| **`asan`** | 启用 AddressSanitizer 与 UndefinedBehaviorSanitizer 内存检测 | Clang 或 GCC Sanitizer 库 |

### 快速编译命令

```bash
# 1. 配置预设（以 release 为例）
cmake --preset release

# 2. 编译全量目标（或指定 -j 并发数）
cmake --build --preset release -j$(nproc)

# 若仅编译基础算子示例：
cmake --build --preset release --target example_basic -j$(nproc)
```

---

## 2. 基础计算图示例运行

项目提供了一批轻量级基础计算图示例（位于 `examples/` 目录下）：

```bash
# 默认在 CPU 后端运行基础算子图：y = (a + b) * c
./build/release/examples/example_basic

# 通过 --device 显式指定硬件加速后端：
./build/release/examples/example_basic --device cpu
./build/release/examples/example_basic --device ispc
./build/release/examples/example_basic --device cuda
./build/release/examples/example_basic --device vulkan
```

---

## 3. LLM 端到端模型推理

VeloMind 提供基于类 LLaMA 架构（SmolLM2 与 TinyLlama）的端到端自回归推理流水线，支持跨硬件后端（CPU / ISPC / CUDA / Vulkan）进行流式文本生成。

### 3.1 模型资产准备

推荐使用官方首选测试模型 **SmolLM2-135M**（权重文件仅约 260 MB，BFloat16），兼顾秒级极速下载与完整的端到端文本推理能力。

运行根目录下的轻量下载脚本（零 Python 依赖，依赖常见 `curl` 或 `wget`）：

```bash
# 1. 默认官方源下载 SmolLM2 到本地 models/ 目录
./scripts/download_model.sh smollm2

# 2. 国内网络环境强烈推荐添加 --hf-mirror 启用高速镜像源
./scripts/download_model.sh smollm2 --hf-mirror

# 3. 亦可下载至用户全局缓存目录（~/.cache/velomind/models/）
./scripts/download_model.sh smollm2 --global-cache --hf-mirror

# 4. 可选下载 TinyLlama-1.1B 模型（约 4.1 GB）
./scripts/download_model.sh tinyllama --hf-mirror

# 5. 或通过 CMake 目标一键触发下载
cmake --build --preset release --target download_smollm2
```

### 3.2 运行交互式聊天控制台（REPL）

模型下载完成后，直接启动对话终端：

```bash
# 启动 SmolLM2 交互控制台（自动探测可用硬件后端）
./build/release/examples/smollm2/smollm2_chat

# 显式指定计算设备
./build/release/examples/smollm2/smollm2_chat --device cpu
./build/release/examples/smollm2/smollm2_chat --device ispc
./build/release/examples/smollm2/smollm2_chat --device cuda
./build/release/examples/smollm2/smollm2_chat --device vulkan

# 单次非交互 Prompt 测试（生成完即退出）
./build/release/examples/smollm2/smollm2_chat --prompt "Once upon a time," --max 64
```

若启动时本地尚未下载模型，控制台在 TTY 终端环境下会友善提示是否自动下载；在非 TTY 环境下则输出清晰的排错指引，不会异常崩溃。

### 3.3 交互式控制台指令

在 `smollm2_chat` 或 `tinyllama_chat` 提示符（`>>>`）下，支持以下斜杠指令：

| 指令 | 说明 |
| :--- | :--- |
| `/help` | 显示帮助信息与当前生成参数（温度、Top-P、最大 Token 数等） |
| `/temp <float>` | 动态调节采样温度（0 为 Greedy 贪婪解码，当前默认 0.7） |
| `/top_p <float>` | 动态调节 Nucleus Top-P 采样阈值（当前默认 0.9） |
| `/max <int>` | 调节单次回答最大生成 Token 数量（当前默认 64） |
| `/system <text>` | 设置或替换前置系统提示词（如 `/system You are a helpful assistant.`） |
| `/system` | 清除当前设置的系统提示词 |
| `/clear` | 清空当前对话上下文历史（保留系统提示词） |
| `/history` | 查看上一轮生成的详细统计数据（Token 数量、耗时、每秒生成吞吐 tok/s） |
| `/model` | 打印当前运行模型的网络超参数配置（隐层大小、头数、层数等） |
| `/save <file>` | 将当前多轮对话历史导出持久化保存至指定文本文件 |
| `/load <file>` | 从保存的文件中恢复多轮对话历史 |
| `/exit` 或 `/quit` | 退出交互式控制台 |

### 3.4 零下载即刻体验（Synthetic 纯管线验证）

在完全离线无网络、或仅用于测试算子图编译与内存规划时，可使用 `--synthetic` 参数以合成权重直接启动推理管线，完全无需外部模型权重：

```bash
./build/release/examples/smollm2/smollm2_chat --synthetic --prompt "Hello"
./build/release/examples/tinyllama/tinyllama_chat --synthetic
```

### 3.5 资产 5 级级联发现机制

推理引擎采用 5 级级联寻址策略探测模型权重和分词器文件：

1. **命令行参数**：`--model <path>` 与 `--tokenizer <path>`（显式指定时若不存在则硬报错拦截，防止混淆）；
2. **环境变量**：优先读取 `VELOMIND_SMOLLM2_DIR` 或 `VELOMIND_TINYLLAMA_DIR`，兜底读取 `VELOMIND_MODEL_DIR`；
3. **仓库根目录**：`<repo_root>/models/SmolLM2-135M/`（或 `TinyLlama_v1.1`）；
4. **用户标准缓存**：`~/.cache/velomind/models/SmolLM2-135M/`（支持 `$XDG_CACHE_HOME`）；
5. **系统兼容已知路径**：系统预设模型目录回退。

---

## 4. 测试与验证

项目通过 Catch2 提供了涵盖算子单元测试、极端形状覆盖、异构跨后端等价性及端到端自回归解码的全量测试套件：

```bash
# 运行当前预设对应的全量测试套件
ctest --preset release --output-on-failure

# 纯 CPU 环境测试
ctest --preset pure-cpu --output-on-failure

# 运行性能微基准评测
./build/release/benchmarks/benchmark_baseline
```
