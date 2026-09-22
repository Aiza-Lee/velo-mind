#pragma once

#include <string>

#include "llama_config.h"

namespace velomind::examples::smollm2 {

using Config = velomind::examples::llama::Config;

inline constexpr Config kSmolLM2_135M = {
    .vocab_size        = 49152,
    .hidden_size       = 576,
    .intermediate_size = 1536,
    .num_layers        = 30,
    .num_heads         = 9,
    .num_kv_heads      = 3,
    .rms_norm_eps      = 1e-5f,
    .rope_theta        = 100000.0f,
    .max_seq_len       = 8192,
    .tie_word_embeddings = true,
};

auto parse_config_json(const std::string& json_text) -> Config;

}
