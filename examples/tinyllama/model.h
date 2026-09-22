#pragma once

#include <cstddef>

#include "llama_config.h"

namespace velomind::examples::tinyllama {

using Config = velomind::examples::llama::Config;

inline constexpr Config kReal = {
    .vocab_size        = 32000,
    .hidden_size       = 2048,
    .intermediate_size = 5632,
    .num_layers        = 22,
    .num_heads         = 32,
    .num_kv_heads      = 4,
    .rms_norm_eps      = 1e-5f,
    .rope_theta        = 10000.0f,
    .max_seq_len       = 2048,
};

inline constexpr Config kTiny = {
    .vocab_size        = 16,
    .hidden_size       = 8,
    .intermediate_size = 16,
    .num_layers        = 1,
    .num_heads         = 2,
    .num_kv_heads      = 2,
    .rms_norm_eps      = 1e-5f,
    .rope_theta        = 10000.0f,
    .max_seq_len       = 32,
};

}
