#include "benchmark_reporter.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>

namespace velomind::benchmark {

namespace fs = std::filesystem;

void BenchmarkReporter::print_header(const SystemMetadata& sys_meta, const BenchmarkRunConfig& cfg) {
    std::cout << std::format(
        "\n============================================================\n"
        " VeloMind 可复现性能基线评测引擎 (Aito Protocol)\n"
        "============================================================\n"
        "  CPU 处理器:   {} ({} 核心)\n"
        "  系统物理内存: {:.1f} GB\n",
        sys_meta.cpu_model, sys_meta.cpu_concurrency, sys_meta.ram_total_gb
    );
    if (!sys_meta.gpu_model.empty()) {
        std::cout << std::format("  GPU 加速卡:   {} ({} GB)\n", sys_meta.gpu_model, sys_meta.gpu_vram_gb);
    }
    std::cout << std::format(
        "  操作系统/内核: {}\n"
        "  编译器/构建:  {} ({})\n"
        "  评测配置参数: 迭代轮数={}, 预热轮数={}, 线程预算={}\n"
        "------------------------------------------------------------\n",
        sys_meta.os_version, sys_meta.compiler, sys_meta.build_type,
        cfg.iterations, cfg.warmup, cfg.threads
    );
}

void BenchmarkReporter::print_row(const BaselineDataPoint& dp) {
    std::string err_desc;
    if (dp.device_str == "CPU") {
        err_desc = "0 (参考)";
    } else if (!dp.error.has_reference) {
        err_desc = "无参考";
    } else {
        err_desc = std::format("{:.2e}", dp.error.max_diff);
    }

    const std::vector<std::string> metrics = {
        std::format("构图中位/P95: {:.2f}/{:.2f} ms", dp.graph_time.median_ms, dp.graph_time.p95_ms),
        std::format("分配: {:.2f} ms", dp.alloc_time.median_ms),
        std::format("Prefill中位/P95: {:.2f}/{:.2f} ms ({:.1f} tok/s)",
                    dp.prefill_time.median_ms, dp.prefill_time.p95_ms, dp.prefill_throughput_tok_s),
        std::format("Decode: {:.2f} ms ({:.1f} tok/s)",
                    dp.decode_time.median_ms, dp.decode_throughput_tok_s),
        std::format("D2H: {:.2f} ms", dp.d2h_time.median_ms),
        std::format("峰值内存: {:.1f} MB", dp.planned_peak_mb),
        std::format("误差: {} (非有限: {})",
                    err_desc, dp.error.all_finite ? "0" : "异常"),
    };

    std::string line = std::format("  [S={:>4}] ", dp.seq_len);
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        line += metrics[i];
        if (i + 1 < metrics.size()) line += " | ";
    }
    line += "\n";
    std::cout << line;
}

void BenchmarkReporter::export_markdown(const std::vector<BaselineDataPoint>& results,
                                        const SystemMetadata& sys_meta,
                                        const BenchmarkRunConfig& cfg,
                                        const std::string& path) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path);
    if (!f) return;

    f << std::format(
        "# VeloMind 可复现性能基线评测报告\n\n"
        "**基准模型:** {} | **构建模式:** {}\n\n"
        "## 1. 硬件与执行环境\n\n"
        "| 项目 | 配置详情 |\n"
        "| :--- | :--- |\n"
        "| **CPU 处理器** | {} ({} 线程) |\n"
        "| **系统物理内存** | {:.1f} GB |\n",
        cfg.model_name, sys_meta.build_type,
        sys_meta.cpu_model, sys_meta.cpu_concurrency,
        sys_meta.ram_total_gb
    );
    if (!sys_meta.gpu_model.empty()) {
        f << std::format("| **GPU 加速卡** | {} ({} GB) |\n", sys_meta.gpu_model, sys_meta.gpu_vram_gb);
    }
    f << std::format(
        "| **操作系统 / 内核** | {} |\n"
        "| **编译器版本** | {} |\n"
        "| **线程预算** | {} 线程 (ISPC 线程池 / OpenMP) |\n"
        "| **评测统计参数** | 迭代 {} 轮，预热 {} 轮，中位数与 P95 |\n\n"
        "## 2. 全生命周期性能基线矩阵\n\n",
        sys_meta.os_version, sys_meta.compiler, cfg.threads, cfg.iterations, cfg.warmup
    );

    // 列定义驱动：表头、对齐线与单元格格式化统一声明，增删列仅需改动一处
    struct MarkdownColumn {
        std::string_view title;
        std::string_view align;
        std::function<std::string(const BaselineDataPoint&)> format_cell;
    };

    static const std::vector<MarkdownColumn> columns = {
        {"后端",              ":---",  [](const BaselineDataPoint& r) { return r.device_str; }},
        {"S",                 ":---:", [](const BaselineDataPoint& r) { return std::to_string(r.seq_len); }},
        {"构图(中位/P95)",    ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f}/{:.1f} ms", r.graph_time.median_ms, r.graph_time.p95_ms); }},
        {"分配(中位/P95)",    ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f}/{:.1f} ms", r.alloc_time.median_ms, r.alloc_time.p95_ms); }},
        {"Prefill(中位/P95)", ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f}/{:.1f} ms", r.prefill_time.median_ms, r.prefill_time.p95_ms); }},
        {"Prefill吞吐",       ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f} tok/s", r.prefill_throughput_tok_s); }},
        {"Decode(中位/P95)",   ":---:", [](const BaselineDataPoint& r) { return std::format("{:.2f}/{:.2f} ms", r.decode_time.median_ms, r.decode_time.p95_ms); }},
        {"Decode吞吐",        ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f} tok/s", r.decode_throughput_tok_s); }},
        {"D2H(中位)",         ":---:", [](const BaselineDataPoint& r) { return std::format("{:.2f} ms", r.d2h_time.median_ms); }},
        {"规划峰值",          ":---:", [](const BaselineDataPoint& r) { return std::format("{:.1f} MB", r.planned_peak_mb); }},
        {"最大误差",          ":---:", [](const BaselineDataPoint& r) {
            if (r.device_str == "CPU") return std::string("0 (参考)");
            if (!r.error.has_reference) return std::string("无参考");
            return std::format("{:.2e}", r.error.max_diff);
        }},
        {"非有限值",          ":---:", [](const BaselineDataPoint& r) { return r.error.all_finite ? "0 (全有限)" : "异常"; }},
    };

    f << "|";
    for (const auto& col : columns) f << " " << col.title << " |";
    f << "\n|";
    for (const auto& col : columns) f << " " << col.align << " |";
    f << "\n";

    for (const auto& r : results) {
        f << "|";
        for (const auto& col : columns) {
            f << " " << col.format_cell(r) << " |";
        }
        f << "\n";
    }
}

} // namespace velomind::benchmark
