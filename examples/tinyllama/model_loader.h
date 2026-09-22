#pragma once

#include <cstddef>

#include "velomind/graph.h"
#include "velomind/safetensors.h"

#include "llama_config.h"
#include "llama_graph.h"

namespace velomind::examples::tinyllama {

auto bind_tinyllama_weights(Graph&        g,
                            const velomind::examples::llama::Config& cfg,
                            std::size_t   seq_len)
    -> velomind::examples::llama::ForwardWeights;

void populate_tinyllama_weights(const SafetensorsFile& sf,
                                const velomind::examples::llama::Config& cfg,
                                velomind::examples::llama::ForwardWeights& w,
                                std::size_t            seq_len);

}
