#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind_core_export.h"

namespace velomind {

class Graph;

// 计算图 DAG 中的操作节点。
// 持有算子描述符（OpDescriptor，含具体算子枚举与静态属性），并记录输入与输出张量存储指针；
// 供 Graph 进行拓扑依赖分析、死代码消除（DCE）以及内存生命周期规划。
class VELOMIND_CORE_EXPORT Node {
public:
    Node() = default;

    auto op()      const -> const OpDescriptor&                  { return _op; }
    auto inputs()  const -> std::span<const pConstTensorStorage> { return _inputs; }
    auto outputs() const -> std::span<const pTensorStorage>      { return _outputs; }

private:
    friend class Graph;
    friend struct GraphTestAccess;

    OpDescriptor                     _op;
    std::vector<pConstTensorStorage> _inputs;
    std::vector<pTensorStorage>      _outputs;

};

VELOMIND_EXPORT_PTR(Node);

} // namespace velomind
