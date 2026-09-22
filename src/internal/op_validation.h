#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include "velomind/ops.h"

namespace velomind::internal {

inline void validate_op_metadata(Op op, std::span<const pConstTensorStorage> in,
                                 const OpAttrs& attrs) {
    const auto fail = [op](const char* reason) {
        throw std::invalid_argument(std::string(op_name(op)) + ": " + reason);
    };
    if (attrs.index() != default_op_attrs(op).index()) fail("invalid attributes");
    const auto& shape = in[0]->shape;
    for (auto* input : in) {
        if (input->shape.empty()) fail("scalar input is not supported");
        for (auto dim : input->shape)
            if (dim == 0) fail("zero dimension is not supported");
    }
    if (op != Op::Embedding && op != Op::MatMul && op != Op::RMSNorm && op != Op::QuantizedMatMul) {
        for (auto* input : in)
            if (input->dtype != in[0]->dtype) fail("input dtype mismatch");
    }
    switch (op) {
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
            if (shape != in[1]->shape) fail("broadcast is not supported; shapes must match");
            break;
        case Op::MatMul: {
            const auto& a = std::get<MatMulAttrs>(attrs);
            if (a.trans_a || a.trans_b) fail("transpose attributes are not supported");
            if (in[0]->dtype != in[1]->dtype) {
                const bool valid_mixed =
                    (in[0]->dtype == DataType::Float32 && (in[1]->dtype == DataType::Float16 || in[1]->dtype == DataType::BFloat16)) ||
                    (in[1]->dtype == DataType::Float32 && (in[0]->dtype == DataType::Float16 || in[0]->dtype == DataType::BFloat16));
                if (!valid_mixed) fail("unsupported mixed precision dtypes for MatMul");
            }
            break;
        }
        case Op::Reshape: {
            TensorStorage target;
            target.shape = std::get<ReshapeAttrs>(attrs).shape;
            if (storage_numel(target) != storage_numel(*in[0])) fail("element count mismatch");
            break;
        }
        case Op::Transpose: {
            const auto& perm = std::get<TransposeAttrs>(attrs).perm;
            if (perm.empty()) break;
            if (perm.size() != shape.size()) fail("permutation rank mismatch");
            std::vector<bool> seen(shape.size());
            for (auto axis : perm) {
                if (axis < 0 || static_cast<std::size_t>(axis) >= shape.size()) fail("permutation axis out of range");
                if (seen[axis]) fail("duplicate permutation axis");
                seen[axis] = true;
            }
            break;
        }
        case Op::Concat: {
            if (shape.size() != in[1]->shape.size()) fail("rank mismatch");
            int64_t axis = std::get<ConcatAttrs>(attrs).axis;
            if (axis < 0) axis += static_cast<int64_t>(shape.size());
            if (axis < 0 || static_cast<std::size_t>(axis) >= shape.size()) fail("axis out of range");
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i != static_cast<std::size_t>(axis) && shape[i] != in[1]->shape[i]) fail("non-axis shape mismatch");
            }
            if (shape[axis] > std::numeric_limits<dim_t>::max() - in[1]->shape[axis]) fail("dimension sum overflow");
            break;
        }
        case Op::RMSNorm: {
            const auto& a = std::get<RMSNormAttrs>(attrs);
            if (shape.empty() || shape.back() == 0) fail("requires a nonempty last dimension");
            int64_t axis = a.axis;
            if (axis < 0) axis += static_cast<int64_t>(shape.size());
            if (axis != static_cast<int64_t>(shape.size()) - 1) fail("only last-axis normalization is supported");
            if (!std::isfinite(a.epsilon) || a.epsilon <= 0) fail("epsilon must be finite and positive");
            if (in[1]->shape != shape_t{shape.back()}) fail("weight shape mismatch");
            const bool d0_ok = in[0]->dtype == DataType::Float32 || in[0]->dtype == DataType::Float16 || in[0]->dtype == DataType::BFloat16;
            if (!d0_ok) fail("expected Float32, Float16, or BFloat16 input for RMSNorm");
            if (in[0]->dtype != in[1]->dtype) {
                const bool valid_mixed =
                    (in[0]->dtype == DataType::Float32 && (in[1]->dtype == DataType::Float16 || in[1]->dtype == DataType::BFloat16));
                if (!valid_mixed) fail("unsupported mixed precision dtypes for RMSNorm");
            }
            break;
        }
        case Op::Embedding:
            if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) fail("expected nonempty rank-two table");
            if (in[0]->dtype != DataType::Float32 || in[1]->dtype != DataType::Int32) fail("expected Float32 table and Int32 indices");
            break;
        case Op::RotaryEmbedding: {
            if (shape.empty() || shape.back() == 0 || shape.back() % 2 != 0) fail("head dimension must be positive and even");
            if (in[0]->dtype != DataType::Float32) fail("expected Float32");
            const auto& cache = in[1]->shape;
            if (cache != in[2]->shape || cache.empty() || cache.size() > 2 || cache.back() != shape.back() / 2)
                fail("cosine/sine cache shape mismatch");
            if (cache.size() == 2 && cache[0] <= 0) fail("empty position cache");
            if (cache.size() == 2 && cache[0] > 1 && (shape.size() < 2 || shape[shape.size() - 2] != cache[0]))
                fail("position cache length mismatch");
            break;
        }
        case Op::RepeatKV: {
            const auto& a = std::get<RepeatKVAttrs>(attrs);
            if (a.repeats <= 0) fail("repeats must be positive");
            if (shape.empty()) fail("cannot repeat scalar");
            int64_t axis = a.axis;
            if (axis < 0) axis += static_cast<int64_t>(shape.size());
            if (axis < 0 || static_cast<std::size_t>(axis) >= shape.size()) fail("axis out of range");
            break;
        }
        case Op::Slice: {
            const auto& a = std::get<SliceAttrs>(attrs);
            if (a.begins.size() != a.ends.size()) fail("begins and ends rank mismatch");
            if (!a.strides.empty() && a.strides.size() != a.begins.size()) fail("strides rank mismatch");
            if (a.begins.size() > shape.size()) fail("slice attributes rank exceeds tensor rank");
            for (std::size_t i = 0; i < a.begins.size(); ++i) {
                if (i < a.strides.size() && a.strides[i] <= 0) fail("stride must be positive");
                const int dim = static_cast<int>(shape[i]);
                int b = a.begins[i] < 0 ? a.begins[i] + dim : a.begins[i];
                int e = a.ends[i] < 0 ? a.ends[i] + dim : a.ends[i];
                b = std::clamp(b, 0, dim);
                e = std::clamp(e, 0, dim);
                if (b >= e) fail("slice begin must be strictly less than end");
            }
            break;
        }
        case Op::FusedAttention: {
            const auto& a = std::get<FusedAttentionAttrs>(attrs);
            if (in[0]->dtype != DataType::Float32 &&
                in[0]->dtype != DataType::Float16 &&
                in[0]->dtype != DataType::BFloat16) {
                fail("expected Float32, Float16, or BFloat16");
            }
            if (in[0]->shape.size() < 3 || in[1]->shape.size() != in[0]->shape.size() || in[2]->shape.size() != in[0]->shape.size()) {
                fail("inputs must have matching rank >= 3");
            }
            const std::size_t rank = in[0]->shape.size();
            for (std::size_t i = 0; i + 3 < rank; ++i) {
                if (in[0]->shape[i] != in[1]->shape[i] || in[0]->shape[i] != in[2]->shape[i]) {
                    fail("batch dimension mismatch");
                }
            }
            if (in[1]->shape[rank - 3] <= 0 || in[2]->shape[rank - 3] != in[1]->shape[rank - 3] ||
                in[0]->shape[rank - 3] % in[1]->shape[rank - 3] != 0) {
                fail("invalid head counts");
            }
            if (in[1]->shape[rank - 2] != in[2]->shape[rank - 2]) {
                fail("K and V sequence length mismatch");
            }
            if (a.is_causal && in[1]->shape[rank - 2] < in[0]->shape[rank - 2]) {
                fail("causal attention requires S_k >= S_q");
            }
            if (in[0]->shape[rank - 1] != in[1]->shape[rank - 1]) {
                fail("Q and K feature dimension mismatch");
            }
            break;
        }
        case Op::QuantizedMatMul: {
            const auto& a = std::get<QuantizedMatMulAttrs>(attrs);
            if (in[0]->dtype != DataType::Float32 &&
                in[0]->dtype != DataType::Float16 &&
                in[0]->dtype != DataType::BFloat16) {
                fail("activation must be Float32, Float16, or BFloat16");
            }
            if (in[1]->dtype != DataType::Int8) {
                fail("quantized weight must be Int8");
            }
            if (in[2]->dtype != DataType::Float32 &&
                in[2]->dtype != DataType::Float16 &&
                in[2]->dtype != DataType::BFloat16) {
                fail("scales must be Float32, Float16, or BFloat16");
            }
            if (in[0]->shape.size() < 2) {
                fail("activation rank must be >= 2");
            }
            if (in[1]->shape.size() != 2) {
                fail("quantized weight must be 2D matrix");
            }
            if (a.quant_type == QuantType::Int4 && a.block_size > 0 && a.block_size % 2 != 0) {
                fail("Int4 block_size must be even");
            }
            break;
        }
        default: break;
    }
}

inline void validate_backend_index_range(Op op, std::span<const pConstTensorStorage> in,
                                         const TensorStorage& out, DeviceType device) {
    std::size_t limit = std::numeric_limits<std::size_t>::max();
    if (device == DeviceType::ISPC) limit = std::numeric_limits<int>::max();
    if (device == DeviceType::VULKAN) limit = std::numeric_limits<std::uint32_t>::max();
    if (limit != std::numeric_limits<std::size_t>::max()) {
        for (auto* tensor : in) {
            if (storage_numel(*tensor) > limit)
                throw std::invalid_argument(std::string(op_name(op)) + ": backend index range exceeded");
            for (auto dim : tensor->shape)
                if (static_cast<std::uint64_t>(dim) > limit)
                    throw std::invalid_argument(std::string(op_name(op)) + ": backend dimension range exceeded");
        }
        if (storage_numel(out) > limit)
            throw std::invalid_argument(std::string(op_name(op)) + ": backend output index range exceeded");
        if (device == DeviceType::ISPC && out.shape.size() > limit)
            throw std::invalid_argument(std::string(op_name(op)) + ": backend rank range exceeded");
    }
    if (device == DeviceType::CUDA && op == Op::MatMul) {
        for (auto* tensor : in)
            for (auto dim : tensor->shape)
                if (dim > std::numeric_limits<int>::max())
                    throw std::invalid_argument("MatMul: cuBLAS dimension range exceeded");
    }
}

} // namespace velomind::internal
