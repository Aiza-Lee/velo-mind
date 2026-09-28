#pragma once

#include <string>
#include <vector>

#include "benchmark_types.h"
#include "system_info.h"

namespace velomind::benchmark {

class BenchmarkReporter {
public:
    static void print_header(const SystemMetadata& sys_meta, const BenchmarkRunConfig& cfg);
    static void print_row(const BaselineDataPoint& dp);

    static void export_markdown(const std::vector<BaselineDataPoint>& results,
                                const SystemMetadata& sys_meta,
                                const BenchmarkRunConfig& cfg,
                                const std::string& path);
};

} // namespace velomind::benchmark
