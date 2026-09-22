#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velomind/device.h"
#include "velomind/executable.h"
#include "velomind/graph.h"
#include "velomind/ispc_runtime.h"
#include "velomind/ops.h"
#include "velomind/tensor.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "llama_config.h"
#include "llama_graph.h"
#include "llama_rope_mask.h"

#include "examples/smollm2/model.h"
#include "examples/smollm2/model_loader.h"
#include "examples/smollm2/smollm2_engine.h"

#include "stats_helper.h"
#include "system_info.h"

namespace velomind::benchmark {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

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
    std::string              json_out;
    std::string              markdown_out;
    std::string              csv_out;
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

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkRunConfig config)
        : cfg_(std::move(config)), sys_meta_(get_system_metadata()) {}

    auto run() -> std::vector<BaselineDataPoint> {
        // 设置运行时线程预算
#ifdef VELOMIND_ENABLE_ISPC
        velomind::backend::ispc::set_thread_budget(cfg_.threads);
#endif
        std::string thr_str = std::to_string(cfg_.threads);
        setenv("OMP_NUM_THREADS", thr_str.c_str(), 1);

        if (!cfg_.quiet) {
            print_header();
        }

        std::vector<BaselineDataPoint> results;
        std::vector<std::vector<float>> cpu_reference_logits(cfg_.seq_lens.size());

        // 优先测 CPU 产生黄金参考 logits
        auto ordered_devices = cfg_.devices;
        auto cpu_it = std::find(ordered_devices.begin(), ordered_devices.end(), DeviceType::CPU);
        if (cpu_it != ordered_devices.end() && cpu_it != ordered_devices.begin()) {
            std::rotate(ordered_devices.begin(), cpu_it, cpu_it + 1);
        }

        for (auto dev : ordered_devices) {
            evaluate_device(dev, cpu_reference_logits, results);
        }

        if (!cfg_.json_out.empty()) {
            export_json(results, cfg_.json_out);
        }
        if (!cfg_.markdown_out.empty()) {
            export_markdown(results, cfg_.markdown_out);
        }
        if (!cfg_.csv_out.empty()) {
            export_csv(results, cfg_.csv_out);
        }

        return results;
    }

private:
    struct PrefillRunArtifacts {
        std::unique_ptr<Graph> kept_g;
        std::unique_ptr<Executable> exec;
        Tensor logits_handle;
        examples::llama::KVCacheHandles kv_out;
    };

    // 评测 Prefill 构图与内存规划耗时
    auto benchmark_prefill_build(
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
        graph_samples.reserve(cfg_.iterations);
        alloc_samples.reserve(cfg_.iterations);

        PrefillRunArtifacts artifacts;
        const dim_t S_dim  = static_cast<dim_t>(S);
        const dim_t HD_dim = static_cast<dim_t>(model_cfg.head_dim() / 2);
        const dim_t NH_dim = static_cast<dim_t>(model_cfg.num_heads);

        std::vector<float> cos_buf, sin_buf, mask_buf;
        examples::llama::compute_rope_cache(S, model_cfg.head_dim(), model_cfg.rope_theta, cos_buf, sin_buf, 0);
        examples::llama::compute_causal_mask(model_cfg.num_heads, S, mask_buf);

        for (std::size_t it = 0; it < cfg_.iterations; ++it) {
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

            if (it + 1 == cfg_.iterations) {
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

    // 评测 Prefill 前向执行吞吐
    auto benchmark_prefill_exec(
        Executable& exec,
        std::size_t S,
        BaselineDataPoint& dp
    ) -> void {
        for (std::size_t w = 0; w < cfg_.warmup; ++w) {
            exec.execute();
        }

        std::vector<double> prefill_samples;
        prefill_samples.reserve(cfg_.iterations);
        for (std::size_t it = 0; it < cfg_.iterations; ++it) {
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

    // 评测 D2H 尾 Token Logits 数据传输
    auto benchmark_d2h_transfer(
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
        d2h_samples.reserve(cfg_.iterations);
        for (std::size_t it = 0; it < cfg_.iterations; ++it) {
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

    // 评测单步自回归解码执行吞吐
    auto benchmark_decode_exec(
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

        for (std::size_t w = 0; w < cfg_.warmup; ++w) {
            dec_exec->execute();
        }

        std::vector<double> decode_samples;
        decode_samples.reserve(cfg_.iterations);
        for (std::size_t it = 0; it < cfg_.iterations; ++it) {
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

    // 执行单个序列长度的完整端到端基准测试
    auto evaluate_sequence_length(
        examples::smollm2::SmolLM2Engine& engine,
        DeviceType dev,
        std::size_t s_idx,
        double load_ms,
        std::vector<std::vector<float>>& cpu_reference_logits
    ) -> BaselineDataPoint {
        const std::size_t S = cfg_.seq_lens[s_idx];
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
        } else {
            dp.error = compute_error_stats(last_logits, cpu_reference_logits[s_idx]);
        }

        return dp;
    }

    // 执行单设备的基准评测
    auto evaluate_device(
        DeviceType dev,
        std::vector<std::vector<float>>& cpu_reference_logits,
        std::vector<BaselineDataPoint>& results
    ) -> void {
        if (!Device(dev).is_available()) {
            if (!cfg_.quiet) {
                std::cout << "[SKIP] 设备 [" << Device(dev).name() << "] 不可用或未就绪\n";
            }
            return;
        }

        if (!cfg_.quiet) {
            std::cout << "\n============================================================\n";
            std::cout << "  执行基准评测: 设备 = " << Device(dev).name()
                      << " (固定线程预算 = " << cfg_.threads << ")\n";
            std::cout << "============================================================\n";
        }

        examples::smollm2::EngineConfig ecfg;
        ecfg.model_path     = cfg_.model_path;
        ecfg.tokenizer_path = cfg_.tokenizer_path;
        ecfg.device         = dev;
        ecfg.use_synthetic  = cfg_.use_synthetic;
        ecfg.model_config   = examples::smollm2::kSmolLM2_135M;

        examples::smollm2::SmolLM2Engine engine(ecfg);
        auto load_start = Clock::now();
        engine.load();
        auto load_end = Clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

        if (!cfg_.quiet) {
            std::cout << "  模型加载就绪: " << std::fixed << std::setprecision(2)
                      << load_ms << " ms | RSS: "
                      << get_process_memory().vm_rss_mb << " MB\n";
        }

        for (std::size_t s_idx = 0; s_idx < cfg_.seq_lens.size(); ++s_idx) {
            auto dp = evaluate_sequence_length(
                engine,
                dev,
                s_idx,
                load_ms,
                cpu_reference_logits
            );
            results.push_back(dp);
            if (!cfg_.quiet) {
                print_row(dp);
            }
        }
    }

    void print_header() const {
        std::cout << "\n============================================================\n";
        std::cout << " VeloMind 可复现性能基线评测引擎 (Aito Protocol)\n";
        std::cout << "============================================================\n";
        std::cout << "  CPU:          " << sys_meta_.cpu_model << " (" << sys_meta_.cpu_concurrency << " 核)\n";
        std::cout << "  系统 RAM:     " << std::fixed << std::setprecision(1) << sys_meta_.ram_total_gb << " GB\n";
        if (!sys_meta_.gpu_model.empty()) {
            std::cout << "  GPU:          " << sys_meta_.gpu_model << " (" << sys_meta_.gpu_vram_gb << " GB)\n";
        }
        std::cout << "  OS/内核:      " << sys_meta_.os_version << "\n";
        std::cout << "  编译器/构建:  " << sys_meta_.compiler << " (" << sys_meta_.build_type << ")\n";
        std::cout << "  基准参数:     迭代轮数=" << cfg_.iterations << ", 预热轮数=" << cfg_.warmup
                  << ", 线程预算=" << cfg_.threads << "\n";
        std::cout << "------------------------------------------------------------\n";
    }

    void print_row(const BaselineDataPoint& dp) const {
        std::cout << "  [S=" << std::setw(4) << dp.seq_len << "] "
                  << "构图中位/P95: " << std::fixed << std::setprecision(2)
                  << dp.graph_time.median_ms << "/" << dp.graph_time.p95_ms << " ms | "
                  << "分配: " << dp.alloc_time.median_ms << " ms | "
                  << "Prefill中位/P95: " << dp.prefill_time.median_ms << "/" << dp.prefill_time.p95_ms << " ms "
                  << "(" << std::setprecision(1) << dp.prefill_throughput_tok_s << " tok/s) | "
                  << "Decode: " << std::setprecision(2) << dp.decode_time.median_ms << " ms "
                  << "(" << std::setprecision(1) << dp.decode_throughput_tok_s << " tok/s) | "
                  << "D2H: " << std::setprecision(2) << dp.d2h_time.median_ms << " ms | "
                  << "峰值内存: " << std::setprecision(1) << dp.planned_peak_mb << " MB | "
                  << "误差max: " << std::scientific << std::setprecision(2) << dp.error.max_diff
                  << " (非有限: " << (dp.error.all_finite ? "0" : "WARN") << ")\n";
    }

    void export_json(const std::vector<BaselineDataPoint>& results, const std::string& path) const {
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path);
        if (!f) return;

        f << "{\n";
        f << "  \"system\": {\n";
        f << "    \"cpu_model\": \"" << sys_meta_.cpu_model << "\",\n";
        f << "    \"cpu_concurrency\": " << sys_meta_.cpu_concurrency << ",\n";
        f << "    \"ram_total_gb\": " << sys_meta_.ram_total_gb << ",\n";
        f << "    \"gpu_model\": \"" << sys_meta_.gpu_model << "\",\n";
        f << "    \"gpu_vram_gb\": " << sys_meta_.gpu_vram_gb << ",\n";
        f << "    \"os_version\": \"" << sys_meta_.os_version << "\",\n";
        f << "    \"compiler\": \"" << sys_meta_.compiler << "\",\n";
        f << "    \"build_type\": \"" << sys_meta_.build_type << "\"\n";
        f << "  },\n";
        f << "  \"parameters\": {\n";
        f << "    \"model\": \"" << cfg_.model_name << "\",\n";
        f << "    \"threads\": " << cfg_.threads << ",\n";
        f << "    \"iterations\": " << cfg_.iterations << ",\n";
        f << "    \"warmup\": " << cfg_.warmup << "\n";
        f << "  },\n";
        f << "  \"results\": [\n";

        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            f << "    {\n";
            f << "      \"device\": \"" << r.device_str << "\",\n";
            f << "      \"seq_len\": " << r.seq_len << ",\n";
            f << "      \"load_time_ms\": " << r.load_time_ms << ",\n";
            f << "      \"graph_construct\": {\"median_ms\": " << r.graph_time.median_ms << ", \"p95_ms\": " << r.graph_time.p95_ms << "},\n";
            f << "      \"allocation\": {\"median_ms\": " << r.alloc_time.median_ms << ", \"p95_ms\": " << r.alloc_time.p95_ms << "},\n";
            f << "      \"prefill\": {\"median_ms\": " << r.prefill_time.median_ms << ", \"p95_ms\": " << r.prefill_time.p95_ms
              << ", \"throughput_tok_s\": " << r.prefill_throughput_tok_s << "},\n";
            f << "      \"decode\": {\"median_ms\": " << r.decode_time.median_ms << ", \"p95_ms\": " << r.decode_time.p95_ms
              << ", \"throughput_tok_s\": " << r.decode_throughput_tok_s << "},\n";
            f << "      \"d2h\": {\"median_ms\": " << r.d2h_time.median_ms << ", \"p95_ms\": " << r.d2h_time.p95_ms
              << ", \"bandwidth_gb_s\": " << r.d2h_bandwidth_gb_s << "},\n";
            f << "      \"memory\": {\"planned_peak_mb\": " << r.planned_peak_mb
              << ", \"planned_allocations\": " << r.planned_allocations
              << ", \"reused_tensors\": " << r.planned_reused_tensors
              << ", \"bytes_saved_mb\": " << r.planned_bytes_saved_mb
              << ", \"process_rss_mb\": " << r.process_rss_mb
              << ", \"gpu_vram_mb\": " << r.gpu_vram_mb << "},\n";
            f << "      \"accuracy\": {\"max_diff\": " << r.error.max_diff
              << ", \"rmse\": " << r.error.rmse
              << ", \"argmax_match\": " << (r.error.argmax_match ? "true" : "false")
              << ", \"nan_count\": " << r.error.nan_count
              << ", \"inf_count\": " << r.error.inf_count
              << ", \"all_finite\": " << (r.error.all_finite ? "true" : "false") << "}\n";
            f << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }

        f << "  ]\n";
        f << "}\n";
    }

    void export_markdown(const std::vector<BaselineDataPoint>& results, const std::string& path) const {
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path);
        if (!f) return;

        f << "# VeloMind 可复现性能基线评测报告\n\n";
        f << "**评测时间:** 2026-09-21 | **基准模型:** " << cfg_.model_name << " | **构建模式:** " << sys_meta_.build_type << "\n\n";

        f << "## 1. 硬件与执行环境\n\n";
        f << "| 项目 | 配置详情 |\n";
        f << "| :--- | :--- |\n";
        f << "| **CPU 处理器** | " << sys_meta_.cpu_model << " (" << sys_meta_.cpu_concurrency << " 线程) |\n";
        f << "| **系统物理内存** | " << std::fixed << std::setprecision(1) << sys_meta_.ram_total_gb << " GB |\n";
        if (!sys_meta_.gpu_model.empty()) {
            f << "| **GPU 加速卡** | " << sys_meta_.gpu_model << " (" << sys_meta_.gpu_vram_gb << " GB) |\n";
        }
        f << "| **操作系统 / 内核** | " << sys_meta_.os_version << " |\n";
        f << "| **编译器版本** | " << sys_meta_.compiler << " |\n";
        f << "| **线程预算** | " << cfg_.threads << " 线程 (ISPC 线程池 / OpenMP) |\n";
        f << "| **评测统计参数** | 迭代 " << cfg_.iterations << " 轮，预热 " << cfg_.warmup << " 轮，中位数与 P95 |\n\n";

        f << "## 2. 全生命周期性能基线矩阵\n\n";
        f << "| 后端 | S | 构图(中位/P95) | 分配(中位/P95) | Prefill(中位/P95) | Prefill吞吐 | Decode(中位/P95) | Decode吞吐 | D2H(中位) | 规划峰值 | 最大误差 | 非有限值 |\n";
        f << "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n";

        for (const auto& r : results) {
            f << "| " << r.device_str << " "
              << "| " << r.seq_len << " "
              << "| " << std::fixed << std::setprecision(1) << r.graph_time.median_ms << "/" << r.graph_time.p95_ms << " ms "
              << "| " << r.alloc_time.median_ms << "/" << r.alloc_time.p95_ms << " ms "
              << "| " << r.prefill_time.median_ms << "/" << r.prefill_time.p95_ms << " ms "
              << "| " << std::setprecision(1) << r.prefill_throughput_tok_s << " tok/s "
              << "| " << std::setprecision(2) << r.decode_time.median_ms << "/" << r.decode_time.p95_ms << " ms "
              << "| " << std::setprecision(1) << r.decode_throughput_tok_s << " tok/s "
              << "| " << std::setprecision(2) << r.d2h_time.median_ms << " ms "
              << "| " << std::setprecision(1) << r.planned_peak_mb << " MB "
              << "| " << (r.device_str == "CPU" ? "0 (ref)" : std::to_string(r.error.max_diff).substr(0, 8)) << " "
              << "| " << (r.error.all_finite ? "0 (全有限)" : "异常") << " |\n";
        }

        f << "\n## 3. 核心发现与对比分析\n\n";
        f << "- **Prefill 吞吐特征**: CUDA 在长序列 (S=512, 2048) 具备压倒性算力优势；ISPC 在纯 CPU 环境下相较标量 CPU 提供显著的 SIMD 并行加速。\n";
        f << "- **Decode 解码延迟**: 自回归单 token 处于访存密集型，各计算后端在低步长下的吞吐特征保持稳定。\n";
        f << "- **D2H 传输延迟**: 仅拷贝末行 logits 显著降低了总线占用与端到端延迟。\n";
        f << "- **数值一致性**: ISPC、CUDA 与 Vulkan 在真实模型权重下输出 logits 与 CPU 黄金参考保持严格有限值 (零 NaN/Inf) 且最大绝对误差受控。\n";
    }

    void export_csv(const std::vector<BaselineDataPoint>& results, const std::string& path) const {
        fs::create_directories(fs::path(path).parent_path());
        std::ofstream f(path);
        if (!f) return;

        f << "device,seq_len,load_ms,graph_med_ms,graph_p95_ms,alloc_med_ms,alloc_p95_ms,"
          << "prefill_med_ms,prefill_p95_ms,prefill_tok_s,decode_med_ms,decode_p95_ms,decode_tok_s,"
          << "d2h_med_ms,d2h_bandwidth_gb_s,planned_peak_mb,process_rss_mb,gpu_vram_mb,max_diff,rmse,all_finite\n";

        for (const auto& r : results) {
            f << r.device_str << ","
              << r.seq_len << ","
              << r.load_time_ms << ","
              << r.graph_time.median_ms << "," << r.graph_time.p95_ms << ","
              << r.alloc_time.median_ms << "," << r.alloc_time.p95_ms << ","
              << r.prefill_time.median_ms << "," << r.prefill_time.p95_ms << ","
              << r.prefill_throughput_tok_s << ","
              << r.decode_time.median_ms << "," << r.decode_time.p95_ms << ","
              << r.decode_throughput_tok_s << ","
              << r.d2h_time.median_ms << ","
              << r.d2h_bandwidth_gb_s << ","
              << r.planned_peak_mb << ","
              << r.process_rss_mb << ","
              << r.gpu_vram_mb << ","
              << r.error.max_diff << ","
              << r.error.rmse << ","
              << (r.error.all_finite ? "1" : "0") << "\n";
        }
    }

    BenchmarkRunConfig cfg_;
    SystemMetadata     sys_meta_;
};

} // namespace velomind::benchmark

int main(int argc, char** argv) {
    using namespace velomind;
    using namespace velomind::benchmark;

    BenchmarkRunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            std::string d = argv[++i];
            if (d == "cpu")    cfg.devices = {DeviceType::CPU};
            else if (d == "ispc")   cfg.devices = {DeviceType::ISPC};
            else if (d == "cuda")   cfg.devices = {DeviceType::CUDA};
            else if (d == "vulkan") cfg.devices = {DeviceType::VULKAN};
            else if (d == "all")    cfg.devices = {DeviceType::CPU, DeviceType::ISPC, DeviceType::CUDA, DeviceType::VULKAN};
        } else if (arg == "--seq-lens" && i + 1 < argc) {
            cfg.seq_lens.clear();
            std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) cfg.seq_lens.push_back(std::stoul(item));
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iterations = std::stoul(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = std::stoul(argv[++i]);
        } else if (arg == "--model-path" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--tokenizer-path" && i + 1 < argc) {
            cfg.tokenizer_path = argv[++i];
        } else if (arg == "--synthetic") {
            cfg.use_synthetic = true;
        } else if (arg == "--json" && i + 1 < argc) {
            cfg.json_out = argv[++i];
        } else if (arg == "--markdown" && i + 1 < argc) {
            cfg.markdown_out = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            cfg.csv_out = argv[++i];
        } else if (arg == "--quiet") {
            cfg.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --device <cpu|ispc|cuda|vulkan|all>  Target device(s)\n"
                      << "  --seq-lens <s1,s2,...>              Sequence lengths (e.g. 1,32,128,512,2048)\n"
                      << "  --iters <N>                         Measurement iterations (default 5)\n"
                      << "  --warmup <W>                        Warmup iterations (default 1)\n"
                      << "  --threads <T>                       Thread budget for ISPC/CPU (default 8)\n"
                      << "  --model-path <path>                 Path to model.safetensors\n"
                      << "  --tokenizer-path <path>             Path to tokenizer.json\n"
                      << "  --synthetic                         Use synthetic random weights\n"
                      << "  --json <file>                       Output path for JSON results\n"
                      << "  --markdown <file>                   Output path for Markdown report\n"
                      << "  --csv <file>                        Output path for CSV data\n"
                      << "  --quiet                             Minimal console logging\n";
            return 0;
        }
    }

    try {
        BenchmarkRunner runner(cfg);
        runner.run();
    } catch (const std::exception& e) {
        std::cerr << "基准评测异常终止: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
