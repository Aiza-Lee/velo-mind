#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <span>

#include "internal/registry/kernel.h"
#include "internal/registry/shape.h"
#include "internal/op_validation.h"

namespace velomind::internal {

auto op_signature(Op                                   op,
                  std::span<const pConstTensorStorage> inputs,
                  const OpAttrs&                       attrs ) -> OpSignature
{
    auto fn = get_op_signature(op);
    if (fn == nullptr) {
        throw std::invalid_argument("Graph::op: unsupported operator signature");
    }
    std::size_t expected = 1;
    switch (op) {
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        case Op::MatMul: case Op::Concat: case Op::RMSNorm: case Op::Embedding:
            expected = 2;
            break;
        case Op::RotaryEmbedding:
        case Op::FusedAttention:
        case Op::QuantizedMatMul:
            expected = 3;
            break;
        default: break;
    }
    if (inputs.size() != expected)
        throw std::invalid_argument(std::string(op_name(op)) + ": invalid input count");
    for (auto* input : inputs) {
        if (!input) throw std::invalid_argument("Graph::op: null input storage");
        if (input->size_bytes < storage_nbytes(*input))
            throw std::invalid_argument("Graph::op: invalid input capacity");
    }
    if (std::holds_alternative<NoAttrs>(attrs)) {
        const auto defaults = default_op_attrs(op);
        validate_op_metadata(op, inputs, defaults);
        return fn(inputs, defaults);
    }
    validate_op_metadata(op, inputs, attrs);
    return fn(inputs, attrs);
}

auto resolve_op_kernel(Op                        op,
                       std::span<const DataType> input_dtypes,
                       DataType                  output_dtype,
                       DeviceType                device      ) -> Executable::KernelFn
{
    if (input_dtypes.size() > 4) return nullptr;

    KernelDtypeKey key;
    key.num_inputs   = input_dtypes.size();
    key.output_dtype = output_dtype;
    for (std::size_t i = 0; i < input_dtypes.size(); ++i) {
        key.input_dtypes[i] = input_dtypes[i];
    }

    auto& entries = op_kernels(device)[static_cast<std::size_t>(op)];
    for (const auto& entry : entries) {
        if (entry.key == key) return entry.fn;
    }
    return nullptr;
}

} // namespace velomind::internal
