#include "internal/registry/shape.h"

namespace velomind {

auto default_op_attrs(Op op) -> OpAttrs {
    const auto idx = static_cast<std::size_t>(op);
    if (idx < MAX_OPS) {
        return internal::default_attrs_table()[idx];
    }
    return NoAttrs{};
}

} // namespace velomind

namespace velomind::internal {

    auto signatures() -> std::array<OpSignatureFn, MAX_OPS>& {
        static std::array<OpSignatureFn, MAX_OPS> r{};
        return r;
    }

    void register_op_signature(Op op, OpSignatureFn fn) {
        signatures()[static_cast<std::size_t>(op)] = fn;
    }

    auto get_op_signature(Op op) -> OpSignatureFn {
        const auto idx = static_cast<std::size_t>(op);
        if (idx >= MAX_OPS) return nullptr;
        return signatures()[idx];
    }

    auto default_attrs_table() -> std::array<OpAttrs, MAX_OPS>& {
        static std::array<OpAttrs, MAX_OPS> table = [] {
            std::array<OpAttrs, MAX_OPS> t{};
            t[static_cast<std::size_t>(Op::NoOp)]            = NoAttrs{};
            t[static_cast<std::size_t>(Op::Add)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Sub)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Mul)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Div)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Neg)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Abs)]             = NoAttrs{};
            t[static_cast<std::size_t>(Op::Relu)]            = NoAttrs{};
            t[static_cast<std::size_t>(Op::Sigmoid)]         = NoAttrs{};
            t[static_cast<std::size_t>(Op::Tanh)]            = NoAttrs{};
            t[static_cast<std::size_t>(Op::Softmax)]         = SoftmaxAttrs{};
            t[static_cast<std::size_t>(Op::MatMul)]          = MatMulAttrs{};
            t[static_cast<std::size_t>(Op::ReduceSum)]       = ReduceAttrs{};
            t[static_cast<std::size_t>(Op::ReduceMean)]      = ReduceAttrs{};
            t[static_cast<std::size_t>(Op::Conv2D)]          = Conv2DAttrs{};
            t[static_cast<std::size_t>(Op::Reshape)]         = ReshapeAttrs{};
            t[static_cast<std::size_t>(Op::Transpose)]       = TransposeAttrs{};
            t[static_cast<std::size_t>(Op::Concat)]          = ConcatAttrs{};
            t[static_cast<std::size_t>(Op::Slice)]           = SliceAttrs{};
            t[static_cast<std::size_t>(Op::Broadcast)]       = NoAttrs{};
            t[static_cast<std::size_t>(Op::Rsqrt)]           = NoAttrs{};
            t[static_cast<std::size_t>(Op::RMSNorm)]         = RMSNormAttrs{};
            t[static_cast<std::size_t>(Op::Embedding)]       = NoAttrs{};
            t[static_cast<std::size_t>(Op::RotaryEmbedding)] = NoAttrs{};
            t[static_cast<std::size_t>(Op::RepeatKV)]        = RepeatKVAttrs{};
            t[static_cast<std::size_t>(Op::FusedAttention)]  = FusedAttentionAttrs{};
            t[static_cast<std::size_t>(Op::QuantizedMatMul)] = QuantizedMatMulAttrs{};
            return t;
        }();
        return table;
    }

    void register_default_op_attrs(Op op, OpAttrs attrs) {
        const auto idx = static_cast<std::size_t>(op);
        if (idx < MAX_OPS) {
            default_attrs_table()[idx] = std::move(attrs);
        }
    }

} // namespace velomind::internal
