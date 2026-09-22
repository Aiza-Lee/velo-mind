#include "llama_rope_mask.h"

#include <cmath>

namespace velomind::examples::llama {

void compute_rope_cache(std::size_t seq_len,
                        std::size_t head_dim,
                        float       base,
                        std::vector<float>& cos_out,
                        std::vector<float>& sin_out,
                        std::size_t pos_offset) {
    const std::size_t half_dim = head_dim / 2;
    cos_out.resize(seq_len * half_dim);
    sin_out.resize(seq_len * half_dim);

    for (std::size_t t = 0; t < seq_len; ++t) {
        for (std::size_t i = 0; i < half_dim; ++i) {
            float exp_val = static_cast<float>(2 * i) / static_cast<float>(head_dim);
            float inv_freq = 1.0f / std::pow(base, exp_val);
            float freq = static_cast<float>(t + pos_offset) * inv_freq;
            cos_out[t * half_dim + i] = std::cos(freq);
            sin_out[t * half_dim + i] = std::sin(freq);
        }
    }
}

void compute_causal_mask(std::size_t num_heads,
                         std::size_t seq_len,
                         std::vector<float>& mask_out) {
    mask_out.resize(num_heads * seq_len * seq_len);
    for (std::size_t h = 0; h < num_heads; ++h) {
        for (std::size_t r = 0; r < seq_len; ++r) {
            for (std::size_t c = 0; c < seq_len; ++c) {
                // 使用有限大负值，避免部分后端在后续运算中传播无穷值。
                float val = (c > r) ? -1e9f : 0.0f;
                mask_out[(h * seq_len + r) * seq_len + c] = val;
            }
        }
    }
}

}
