#include "benchmark_runner.h"
#include "benchmark_reporter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <span>

#include "velomind/ispc_runtime.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"

#include "llama_rope_mask.h"

namespace velomind::benchmark {

using Clock = std::chrono::steady_clock;

BenchmarkRunner::BenchmarkRunner(BenchmarkRunConfig config)
    : _cfg(std::move(config)), _sys_meta(get_system_metadata()) {}

auto BenchmarkRunner::run() -> std::vector<BaselineDataPoint> {
#ifdef VELOMIND_ENABLE_ISPC
    velomind::backend::ispc::set_thread_budget(_cfg.threads);
#endif
    std::string thr_str = std::to_string(_cfg.threads);
    setenv("OMP_NUM_THREADS", thr_str.c_str(), 1);

    if (!_cfg.quiet) {
        BenchmarkReporter::print_header(_sys_meta, _cfg);
    }

    std::vector<BaselineDataPoint> results;
    std::vector<std::vector<float>> cpu_reference_logits(_cfg.seq_lens.size());

    // 优先评测 CPU 获取作为数值对比的黄金参考 logits
    auto ordered_devices = _cfg.devices;
    auto cpu_it = std::find(ordered_devices.begin(), ordered_devices.end(), DeviceType::CPU);
    if (cpu_it != ordered_devices.end() && cpu_it != ordered_devices.begin()) {
        std::rotate(ordered_devices.begin(), cpu_it, cpu_it + 1);
    }

    for (auto dev : ordered_devices) {
        evaluate_device(dev, cpu_reference_logits, results);
    }

    if (!_cfg.markdown_out.empty()) {
        BenchmarkReporter::export_markdown(results, _sys_meta, _cfg, _cfg.markdown_out);
    }

    return results;
}

auto BenchmarkRunner::benchmark_prefill_build(
    examples::smollm2::SmolLM2Engine& engine,
    DeviceType dev,
    std::size_t S,
    BaselineDataPoint& dp
) -> PrefillRunArtifacts {
    const auto& model_cfg = engine.config().model_config;
    std::vector<std::int32_t> tokens(S);
    for (std::size_t i = 0; i < S; ++i) {
        tokens[i] = static_cast<std::int32_t>((i % 1000) + 1);
    }

    std::vector<double> graph_samples;
    std::vector<double> alloc_samples;
    graph_samples.reserve(_cfg.iterations);
    alloc_samples.reserve(_cfg.iterations);

    PrefillRunArtifacts artifacts;
    const dim_t S_dim  = static_cast<dim_t>(S);
    const dim_t HD_dim = static_cast<dim_t>(model_cfg.head_dim() / 2);
    const dim_t NH_dim = static_cast<dim_t>(model_cfg.num_heads);

    std::vector<float> cos_buf, sin_buf, mask_buf;
    examples::llama::compute_rope_cache(S, model_cfg.head_dim(), model_cfg.rope_theta, cos_buf, sin_buf, 0);
    examples::llama::compute_causal_mask(model_cfg.num_heads, S, mask_buf);

    for (std::size_t it = 0; it < _cfg.iterations; ++it) {
        auto g_start = Clock::now();
        auto g_ptr = std::make_unique<Graph>();
        auto tokens_t = g_ptr->input({static_cast<dim_t>(S)}, DataType::Int32);
        examples::llama::ForwardWeights step_w;
        engine.inner().bind_model_weights(*g_ptr, step_w);

        step_w.cos_cache   = g_ptr->input({S_dim, HD_dim}, DataType::Float32);
        step_w.sin_cache   = g_ptr->input({S_dim, HD_dim}, DataType::Float32);
        step_w.causal_mask = g_ptr->input({NH_dim, S_dim, S_dim}, DataType::Float32);

        examples::llama::KVCacheHandles kv_local;
        auto logits_t = examples::llama::build_forward_graph_prefill(
            *g_ptr, model_cfg, tokens_t, step_w, &kv_local, true
        );
        auto g_end = Clock::now();
        graph_samples.push_back(
            std::chrono::duration<double, std::milli>(g_end - g_start).count()
        );

        auto a_start = Clock::now();
        auto cur_exec = g_ptr->build(dev);
        auto a_end = Clock::now();
        alloc_samples.push_back(
            std::chrono::duration<double, std::milli>(a_end - a_start).count()
        );

        if (it + 1 == _cfg.iterations) {
            tokens_t.copy_from_host(std::as_bytes(std::span(tokens)));
            step_w.cos_cache.copy_from_host(std::as_bytes(std::span(cos_buf)));
            step_w.sin_cache.copy_from_host(std::as_bytes(std::span(sin_buf)));
            step_w.causal_mask.copy_from_host(std::as_bytes(std::span(mask_buf)));

            artifacts.kept_g = std::move(g_ptr);
            artifacts.exec = std::move(cur_exec);
            artifacts.logits_handle = logits_t;
            artifacts.kv_out = std::move(kv_local);

            const auto& pstats = artifacts.exec->memory_plan_stats();
            dp.planned_peak_mb        = static_cast<double>(pstats.peak_bytes) / (1024.0 * 1024.0);
            dp.planned_allocations    = pstats.allocation_count;
            dp.planned_reused_tensors = pstats.reused_tensor_count;
            dp.planned_bytes_saved_mb = static_cast<double>(pstats.bytes_saved) / (1024.0 * 1024.0);
        }
    }

    dp.graph_time = compute_timing_stats(graph_samples);
    dp.alloc_time = compute_timing_stats(alloc_samples);
    return artifacts;
}

auto BenchmarkRunner::benchmark_prefill_exec(
    Executable& exec,
    std::size_t S,
    BaselineDataPoint& dp
) -> void {
    for (std::size_t w = 0; w < _cfg.warmup; ++w) {
        exec.execute();
    }

    std::vector<double> prefill_samples;
    prefill_samples.reserve(_cfg.iterations);
    for (std::size_t it = 0; it < _cfg.iterations; ++it) {
        auto t0 = Clock::now();
        exec.execute();
        auto t1 = Clock::now();
        prefill_samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }
    dp.prefill_time = compute_timing_stats(prefill_samples);
    dp.prefill_throughput_tok_s =
        (dp.prefill_time.median_ms > 0.0)
            ? (static_cast<double>(S) / (dp.prefill_time.median_ms / 1000.0))
            : 0.0;
}

auto BenchmarkRunner::benchmark_d2h_transfer(
    Tensor& logits_handle,
    std::size_t vocab_size,
    std::vector<float>& last_logits,
    BaselineDataPoint& dp
) -> void {
    last_logits.resize(vocab_size);
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(last_logits.data()),
        last_logits.size() * sizeof(float)
    );

    std::vector<double> d2h_samples;
    d2h_samples.reserve(_cfg.iterations);
    for (std::size_t it = 0; it < _cfg.iterations; ++it) {
        auto t0 = Clock::now();
        logits_handle.copy_to_host(byte_span);
        auto t1 = Clock::now();
        d2h_samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }
    dp.d2h_time = compute_timing_stats(d2h_samples);
    const double d2h_bytes = static_cast<double>(vocab_size * sizeof(float));
    dp.d2h_bandwidth_gb_s =
        (dp.d2h_time.median_ms > 0.0)
            ? (d2h_bytes / (dp.d2h_time.median_ms * 1e6))
            : 0.0;
}

auto BenchmarkRunner::benchmark_decode_exec(
    examples::smollm2::SmolLM2Engine& engine,
    DeviceType dev,
    std::size_t S,
    const examples::llama::KVCacheHandles& kv_out,
    BaselineDataPoint& dp
) -> void {
    const auto& model_cfg = engine.config().model_config;
    const dim_t HD_dim = static_cast<dim_t>(model_cfg.head_dim() / 2);

    Graph dec_g;
    auto dec_token_t = dec_g.input({1}, DataType::Int32);
    examples::llama::ForwardWeights dec_w;
    engine.inner().bind_model_weights(dec_g, dec_w);

    dec_w.cos_cache = dec_g.input({1, HD_dim}, DataType::Float32);
    dec_w.sin_cache = dec_g.input({1, HD_dim}, DataType::Float32);

    examples::llama::KVCacheHandles kv_in;
    kv_in.k.reserve(model_cfg.num_layers);
    kv_in.v.reserve(model_cfg.num_layers);
    for (std::size_t l = 0; l < model_cfg.num_layers; ++l) {
        kv_in.k.push_back(dec_g.input(kv_out.k[l].shared_storage()));
        kv_in.v.push_back(dec_g.input(kv_out.v[l].shared_storage()));
    }

    examples::llama::KVCacheHandles dec_kv_out;
    auto dec_logits_t = examples::llama::build_forward_graph_decode(
        dec_g, model_cfg, dec_token_t, dec_w, kv_in, &dec_kv_out
    );
    auto dec_exec = dec_g.build(dev);

    std::array<std::int32_t, 1> next_tok = {10};
    dec_token_t.copy_from_host(std::as_bytes(std::span(next_tok)));

    std::vector<float> d_cos, d_sin;
    examples::llama::compute_rope_cache(1, model_cfg.head_dim(), model_cfg.rope_theta, d_cos, d_sin, S);
    dec_w.cos_cache.copy_from_host(std::as_bytes(std::span(d_cos)));
    dec_w.sin_cache.copy_from_host(std::as_bytes(std::span(d_sin)));

    for (std::size_t w = 0; w < _cfg.warmup; ++w) {
        dec_exec->execute();
    }

    std::vector<double> decode_samples;
    decode_samples.reserve(_cfg.iterations);
    for (std::size_t it = 0; it < _cfg.iterations; ++it) {
        auto t0 = Clock::now();
        dec_exec->execute();
        auto t1 = Clock::now();
        decode_samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }
    dp.decode_time = compute_timing_stats(decode_samples);
    dp.decode_throughput_tok_s =
        (dp.decode_time.median_ms > 0.0)
            ? (1000.0 / dp.decode_time.median_ms)
            : 0.0;
}

auto BenchmarkRunner::evaluate_sequence_length(
    examples::smollm2::SmolLM2Engine& engine,
    DeviceType dev,
    std::size_t s_idx,
    double load_ms,
    std::vector<std::vector<float>>& cpu_reference_logits
) -> BaselineDataPoint {
    const std::size_t S = _cfg.seq_lens[s_idx];
    BaselineDataPoint dp;
    dp.device_str   = Device(dev).name();
    dp.seq_len      = S;
    dp.load_time_ms = load_ms;

    auto artifacts = benchmark_prefill_build(engine, dev, S, dp);
    benchmark_prefill_exec(*artifacts.exec, S, dp);

    const auto& model_cfg = engine.config().model_config;
    std::vector<float> last_logits;
    benchmark_d2h_transfer(artifacts.logits_handle, model_cfg.vocab_size, last_logits, dp);
    benchmark_decode_exec(engine, dev, S, artifacts.kv_out, dp);

    auto proc_mem = get_process_memory();
    dp.process_rss_mb = proc_mem.vm_hwm_mb > 0.0 ? proc_mem.vm_hwm_mb : proc_mem.vm_rss_mb;
    if (dev == DeviceType::CUDA) {
        dp.gpu_vram_mb = get_cuda_memory().used_mb;
    }

    if (dev == DeviceType::CPU) {
        cpu_reference_logits[s_idx] = last_logits;
        dp.error = compute_error_stats(last_logits, last_logits);
        dp.error.has_reference = true;
    } else if (!cpu_reference_logits[s_idx].empty()) {
        dp.error = compute_error_stats(last_logits, cpu_reference_logits[s_idx]);
    } else {
        dp.error = compute_error_stats(last_logits, {});
        dp.error.has_reference = false;
    }

    return dp;
}

auto BenchmarkRunner::evaluate_device(
    DeviceType dev,
    std::vector<std::vector<float>>& cpu_reference_logits,
    std::vector<BaselineDataPoint>& results
) -> void {
    if (!Device(dev).is_available()) {
        if (!_cfg.quiet) {
            std::cout << std::format("[跳过] 设备 [{}] 不可用或未就绪\n", Device(dev).name());
        }
        return;
    }

    if (!_cfg.quiet) {
        std::cout << std::format(
            "\n============================================================\n"
            "  执行基准评测: 设备 = {} (固定线程预算 = {})\n"
            "============================================================\n",
            Device(dev).name(), _cfg.threads
        );
    }

    examples::smollm2::EngineConfig ecfg;
    ecfg.model_path     = _cfg.model_path;
    ecfg.tokenizer_path = _cfg.tokenizer_path;
    ecfg.device         = dev;
    ecfg.use_synthetic  = _cfg.use_synthetic;
    ecfg.model_config   = examples::smollm2::kSmolLM2_135M;

    examples::smollm2::SmolLM2Engine engine(ecfg);
    auto load_start = Clock::now();
    engine.load();
    auto load_end = Clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    if (!_cfg.quiet) {
        std::cout << std::format(
            "  模型加载就绪: {:.2f} ms | 物理内存 RSS: {:.2f} MB\n",
            load_ms, get_process_memory().vm_rss_mb
        );
    }

    for (std::size_t s_idx = 0; s_idx < _cfg.seq_lens.size(); ++s_idx) {
        auto dp = evaluate_sequence_length(
            engine,
            dev,
            s_idx,
            load_ms,
            cpu_reference_logits
        );
        results.push_back(dp);
        if (!_cfg.quiet) {
            BenchmarkReporter::print_row(dp);
        }
    }
}

} // namespace velomind::benchmark
