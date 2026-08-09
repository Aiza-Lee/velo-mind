#include "velomind/tensor.h"

namespace velomind::ops {

static inline void add(const Tensor& a, const Tensor& b, Tensor& result) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch for addition");
    }
    
}
    
} // namespace velomind
