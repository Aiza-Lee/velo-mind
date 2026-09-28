#pragma once

#include <memory>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"

namespace velomind::internal {

// 计算图编译器：负责 DAG 拓扑分析、张量生命周期分析、别名规划、连续 Arena 内存打包与物理分配及 Executable 执行计划装配。
class GraphCompiler {
public:
    // 将已定义的计算图编译为指定设备上的可执行计划。
    static auto compile(Graph& graph, DeviceType device) -> std::unique_ptr<Executable>;
};

} // namespace velomind::internal
