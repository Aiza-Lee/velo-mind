#pragma once

#include <cstddef>

namespace velomind::examples::llama {

struct Config {
    std::size_t vocab_size       = 0;
    std::size_t hidden_size      = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_layers       = 0;
    std::size_t num_heads        = 0;
    std::size_t num_kv_heads     = 0;
    std::size_t effective_num_kv_heads() const noexcept {
        return num_kv_heads > 0 ? num_kv_heads : num_heads;
    }
    std::size_t head_dim() const noexcept {
        return num_heads == 0 ? 0 : hidden_size / num_heads;
    }
    float       rms_norm_eps    = 1e-5f;
    float       rope_theta      = 10000.0f;
    std::size_t max_seq_len     = 2048;
    bool        tie_word_embeddings = false;
};

}
