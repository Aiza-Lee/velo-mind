#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "velomind/quant.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind {

class Tensor;
class Executable;

using KernelFn = void (*)(const pConstTensorStorage* inputs,
                          const pTensorStorage*      outputs,
                          const void*                attrs);

// 枚举值必须连续；Last 用于确定每设备算子注册表大小。
enum class Op : std::uint16_t {
    NoOp = 0,

    Add,
    Sub,
    Mul,
    Div,

    Neg,
    Abs,

    Relu,
    Sigmoid,
    Tanh,
    Softmax,

    MatMul,

    ReduceSum,
    ReduceMean,

    Conv2D,

    Reshape,
    Transpose,
    Concat,
    Slice,
    Broadcast,

    /* LLM / Transformer */
    Rsqrt,
    RMSNorm,
    Embedding,
    RotaryEmbedding,
    RepeatKV,
    FusedAttention,
    QuantizedMatMul,

    Last = QuantizedMatMul,
};

constexpr std::size_t MAX_OPS = static_cast<std::size_t>(Op::Last) + 1;

struct NoAttrs {};

struct BinaryAttrs {};

struct UnaryAttrs {};

struct MatMulAttrs {
    bool trans_a = false;
    bool trans_b = false;
};

struct ReduceAttrs {
    std::vector<int> axes;
    bool keepdim = false;
};

struct Conv2DAttrs {
    std::array<int, 2> padding {0, 0};
    std::array<int, 2> stride  {1, 1};
    std::array<int, 2> dilation{1, 1};
    int groups = 1;
};

struct SoftmaxAttrs {
    // Float32 连续张量沿此轴归一化；负轴从末维计数，拒绝标量与非正维度。
    int axis = -1;
};

struct ReshapeAttrs {
    shape_t shape;
};

struct RMSNormAttrs {
    float epsilon = 1e-5f;
    int   axis    = -1;
};

struct TransposeAttrs {
    std::vector<int> perm;
};

struct ConcatAttrs {
    int axis = 0;
};

struct SliceAttrs {
    std::vector<int> begins;
    std::vector<int> ends;
    std::vector<int> strides;
};

struct RepeatKVAttrs {
    int repeats = 1;
    int axis    = 0;
};

struct FusedAttentionAttrs {
    // 缩放因子；<= 0.0f 时默认计算为 1.0f / sqrt(head_dim)
    float scale     = 1.0f;
    bool  is_causal = true;
};

struct QuantizedMatMulAttrs {
    QuantType quant_type = QuantType::Int8;
    int       block_size = 0; // 0 表示按通道量化，>0 表示分块量化
    bool      trans_b    = false;
};

using OpAttrs = std::variant<
    NoAttrs,
    BinaryAttrs,
    UnaryAttrs,
    MatMulAttrs,
    ReduceAttrs,
    Conv2DAttrs,
    SoftmaxAttrs,
    ReshapeAttrs,
    TransposeAttrs,
    ConcatAttrs,
    SliceAttrs,
    RMSNormAttrs,
    RepeatKVAttrs,
    FusedAttentionAttrs,
    QuantizedMatMulAttrs
>;

// 算子描述符：绑定具体算子种类及其强类型静态属性，作为 Graph 节点与 Executable 编译调度的元数据载体
struct OpDescriptor {
    Op      op    = Op::NoOp;
    OpAttrs attrs = NoAttrs{};
};

// 获取特定算子的默认属性配置
auto default_op_attrs(Op op) -> OpAttrs;

constexpr const char* op_name(Op op) {
    switch (op) {
        case Op::NoOp:            return "NoOp";
        case Op::Add:             return "Add";
        case Op::Sub:             return "Sub";
        case Op::Mul:             return "Mul";
        case Op::Div:             return "Div";
        case Op::Neg:             return "Neg";
        case Op::Abs:             return "Abs";
        case Op::Relu:            return "Relu";
        case Op::Sigmoid:         return "Sigmoid";
        case Op::Tanh:            return "Tanh";
        case Op::Softmax:         return "Softmax";
        case Op::MatMul:          return "MatMul";
        case Op::ReduceSum:       return "ReduceSum";
        case Op::ReduceMean:      return "ReduceMean";
        case Op::Conv2D:          return "Conv2D";
        case Op::Reshape:         return "Reshape";
        case Op::Transpose:       return "Transpose";
        case Op::Concat:          return "Concat";
        case Op::Slice:           return "Slice";
        case Op::Broadcast:       return "Broadcast";
        case Op::Rsqrt:           return "Rsqrt";
        case Op::RMSNorm:         return "RMSNorm";
        case Op::Embedding:       return "Embedding";
        case Op::RotaryEmbedding: return "RotaryEmbedding";
        case Op::RepeatKV:        return "RepeatKV";
        case Op::FusedAttention:  return "FusedAttention";
        case Op::QuantizedMatMul: return "QuantizedMatMul";
    }
    return "Unknown";
}

inline auto to_string(Op op) -> std::string {
    return op_name(op);
}

inline auto operator<<(std::ostream& os, Op op) -> std::ostream& {
    return os << op_name(op);
}

inline auto to_string(const OpDescriptor& desc) -> std::string {
    return std::string("OpDescriptor(op=") + op_name(desc.op) + ")";
}

inline auto operator<<(std::ostream& os, const OpDescriptor& desc) -> std::ostream& {
    return os << to_string(desc);
}

} // namespace velomind
