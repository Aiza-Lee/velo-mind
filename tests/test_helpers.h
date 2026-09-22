#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "internal/log.h"

namespace velomind_test {

template <typename T>
inline std::span<const std::byte> as_bytes(const std::vector<T>& v) {
    return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(v.data()),
        v.size() * sizeof(T));
}

template <typename T>
inline std::span<std::byte> as_writeable_bytes(std::vector<T>& v) {
    return std::span<std::byte>(
        reinterpret_cast<std::byte*>(v.data()),
        v.size() * sizeof(T));
}

// 测试期间压制框架内部日志，防止预期失败用例污染标准错误输出
struct ScopedLogSilencer {
    velomind::internal::LogHandler prev;
    ScopedLogSilencer() noexcept {
        prev = velomind::internal::get_log_handler();
        velomind::internal::set_log_handler([](velomind::internal::LogLevel, std::string_view) {});
    }
    ~ScopedLogSilencer() {
        velomind::internal::set_log_handler(prev);
    }
};

}

#include "velomind/graph.h"
#include "velomind/node.h"

namespace velomind {

struct GraphTestAccess {
    static auto get_nodes(Graph& g) -> std::vector<Node>& {
        return g._nodes;
    }
    static void set_node_inputs(Node& n, std::vector<pConstTensorStorage> inputs) {
        n._inputs = std::move(inputs);
    }
    static auto get_topological_order(const Graph& g) -> std::vector<pConstNode> {
        return g._topological_order();
    }
};

}
