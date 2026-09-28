#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "velomind/types.h"
#include "stats_helper.h"

namespace velomind::benchmark {

struct BenchmarkRunConfig {
    std::string              model_name   = "smollm2";
    std::string              model_path   = "/home/aiza/workspace/assets/ai-models/SmolLM2-135M/model.safetensors";
    std::string              tokenizer_path = "/home/aiza/workspace/assets/ai-models/SmolLM2-135M/tokenizer.json";
    std::vector<DeviceType>  devices      = {DeviceType::CPU, DeviceType::ISPC, DeviceType::CUDA, DeviceType::VULKAN};
    std::vector<std::size_t> seq_lens     = {1, 32, 128, 512, 2048};
    std::size_t              decode_steps = 4;
    std::size_t              iterations   = 5;
    std::size_t              warmup       = 1;
    std::size_t              threads      = 8;
    bool                     use_synthetic = false;
    std::string              markdown_out;
    bool                     quiet        = false;
};

struct BaselineDataPoint {
    std::string device_str;
    std::size_t seq_len = 0;

    // 各阶段延迟（毫秒）
    double      load_time_ms = 0.0;
    TimingStats graph_time;
    TimingStats alloc_time;
    TimingStats prefill_time;
    TimingStats d2h_time;
    TimingStats decode_time;

    // 吞吐与带宽
    double      prefill_throughput_tok_s = 0.0;
    double      decode_throughput_tok_s  = 0.0;
    double      d2h_bandwidth_gb_s       = 0.0;

    // 内存指标
    double      planned_peak_mb          = 0.0;
    std::size_t planned_allocations      = 0;
    std::size_t planned_reused_tensors   = 0;
    double      planned_bytes_saved_mb   = 0.0;
    double      process_rss_mb           = 0.0;
    double      gpu_vram_mb              = 0.0;

    // 数值与误差
    ErrorStats  error;
};

} // namespace velomind::benchmark
