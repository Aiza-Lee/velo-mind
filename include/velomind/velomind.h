#pragma once

// VeloMind 全局宏定义与导出控制
#include "velomind/macros.h"

// 核心基础类型、设备与数据类型体系
#include "velomind/types.h"
#include "velomind/device.h"
#include "velomind/dtype.h"
#include "velomind/ptr.h"

// 存储底座与张量视图句柄
#include "velomind/tensor_storage.h"
#include "velomind/tensor.h"

// 算子元数据、节点定义与计算图运行时
#include "velomind/ops.h"
#include "velomind/node.h"
#include "velomind/executable.h"
#include "velomind/graph.h"

// 权重持久化格式与量化工具集
#include "velomind/safetensors.h"
#include "velomind/quant.h"

// ISPC 高性能并行运行时
#include "velomind/ispc_runtime.h"
