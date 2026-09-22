#pragma once

#include <cstddef>
#include <vector>

namespace velomind::examples::llama {

void compute_rope_cache(std::size_t seq_len,
                        std::size_t head_dim,
                        float       base,
                        std::vector<float>& cos_out,
                        std::vector<float>& sin_out,
                        std::size_t pos_offset = 0);

void compute_causal_mask(std::size_t num_heads,
                         std::size_t seq_len,
                         std::vector<float>& mask_out);

}
