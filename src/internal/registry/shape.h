#pragma once

#include <array>
#include <cstddef>
#include <span>

#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind::internal {

    struct OpSignature {
        DataType dtype;
        shape_t  shape;
    };

    using OpSignatureFn = OpSignature (*)(std::span<const pConstTensorStorage> inputs,
                                        const OpAttrs&                       attrs);

    auto op_signature(Op                                   op,
                    std::span<const pConstTensorStorage> inputs,
                    const OpAttrs&                       attrs) -> OpSignature;

    auto signatures() -> std::array<OpSignatureFn, MAX_OPS>&;

    void register_op_signature(Op op, OpSignatureFn fn);

    auto get_op_signature(Op op) -> OpSignatureFn;

    auto default_attrs_table() -> std::array<OpAttrs, MAX_OPS>&;

    void register_default_op_attrs(Op op, OpAttrs attrs);

    using ::velomind::default_op_attrs;

} // namespace velomind::internal

#define VELOMIND_CONCAT_INNER_(a, b) a##b
#define VELOMIND_CONCAT_(a, b) VELOMIND_CONCAT_INNER_(a, b)

#define VELOMIND_REGISTER_SIGNATURE_IMPL(OP_ENUM, FN, suffix)                                  \
    namespace {                                                                                \
        [[maybe_unused]] const auto VELOMIND_CONCAT_(_velomind_sig_register_, suffix) = [] {   \
            ::velomind::internal::register_op_signature(OP_ENUM, FN);                          \
            return true;                                                                       \
        }();                                                                                   \
    }

#define VELOMIND_REGISTER_SIGNATURE(OP_ENUM, FN)                                               \
    VELOMIND_REGISTER_SIGNATURE_IMPL(OP_ENUM, FN, __LINE__)
