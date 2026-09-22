#pragma once

#include "velomind/graph.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "llama_config.h"
#include "llama_graph.h"

namespace velomind::examples::tinyllama {

auto build_forward_graph(Graph&           g,
                         const velomind::examples::llama::Config& cfg,
                         Tensor           token,
                         const velomind::examples::llama::ForwardWeights& w) -> Tensor;

}
