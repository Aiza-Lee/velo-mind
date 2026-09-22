#pragma once

#include <limits>
#include <stdexcept>
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"

namespace velomind::detail {

struct SoftmaxLayout {
    std::size_t outer = 1;
    std::size_t cols = 1;
    std::size_t inner = 1;
};

inline auto softmax_layout(const shape_t& shape, const SoftmaxAttrs& attrs) -> SoftmaxLayout {
    const auto rank = static_cast<int64_t>(shape.size());
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("Softmax: axis out of range");
    SoftmaxLayout layout;
    std::size_t total = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) throw std::invalid_argument("Softmax: dimensions must be positive");
        const auto dim = static_cast<std::size_t>(shape[i]);
        if (total > std::numeric_limits<std::size_t>::max() / dim)
            throw std::invalid_argument("Softmax: element count overflow");
        total *= dim;
        if (i < static_cast<std::size_t>(axis)) layout.outer *= dim;
        else if (i == static_cast<std::size_t>(axis)) layout.cols = dim;
        else layout.inner *= dim;
    }
    return layout;
}

inline auto softmax_layout(const TensorStorage& input, const void* descriptor) -> SoftmaxLayout {
    return softmax_layout(input.shape, std::get<SoftmaxAttrs>(
        static_cast<const OpDescriptor*>(descriptor)->attrs));
}

} // namespace velomind::detail
