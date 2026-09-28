#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/tensor.h"
#include "velomind/types.h"

#include "llama_config.h"
#include "llama_graph.h"
#include "examples/smollm2/smollm2_engine.h"

#include "benchmark_types.h"
#include "system_info.h"

namespace velomind::benchmark {

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkRunConfig config);

    auto run() -> std::vector<BaselineDataPoint>;

private:
    struct PrefillRunArtifacts {
        std::unique_ptr<Graph> kept_g;
        std::unique_ptr<Executable> exec;
        Tensor logits_handle;
        examples::llama::KVCacheHandles kv_out;
    };

    auto benchmark_prefill_build(
        examples::smollm2::SmolLM2Engine& engine,
        DeviceType dev,
        std::size_t S,
        BaselineDataPoint& dp
    ) -> PrefillRunArtifacts;

    auto benchmark_prefill_exec(
        Executable& exec,
        std::size_t S,
        BaselineDataPoint& dp
    ) -> void;

    auto benchmark_d2h_transfer(
        Tensor& logits_handle,
        std::size_t vocab_size,
        std::vector<float>& last_logits,
        BaselineDataPoint& dp
    ) -> void;

    auto benchmark_decode_exec(
        examples::smollm2::SmolLM2Engine& engine,
        DeviceType dev,
        std::size_t S,
        const examples::llama::KVCacheHandles& kv_out,
        BaselineDataPoint& dp
    ) -> void;

    auto evaluate_sequence_length(
        examples::smollm2::SmolLM2Engine& engine,
        DeviceType dev,
        std::size_t s_idx,
        double load_ms,
        std::vector<std::vector<float>>& cpu_reference_logits
    ) -> BaselineDataPoint;

    auto evaluate_device(
        DeviceType dev,
        std::vector<std::vector<float>>& cpu_reference_logits,
        std::vector<BaselineDataPoint>& results
    ) -> void;

    BenchmarkRunConfig _cfg;
    SystemMetadata     _sys_meta;
};

} // namespace velomind::benchmark
