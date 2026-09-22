#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace velomind::benchmark {

struct TimingStats {
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double mean_ms   = 0.0;
    double median_ms = 0.0;
    double p95_ms    = 0.0;
};

inline auto compute_timing_stats(std::vector<double> samples) -> TimingStats {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    TimingStats s;
    s.min_ms = samples.front();
    s.max_ms = samples.back();

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean_ms = sum / static_cast<double>(samples.size());

    const std::size_t n = samples.size();
    if (n % 2 == 1) {
        s.median_ms = samples[n / 2];
    } else {
        s.median_ms = 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    const std::size_t p95_idx = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
    s.p95_ms = samples[std::min(p95_idx, n - 1)];
    return s;
}

struct ErrorStats {
    float       max_diff       = 0.0f;
    double      rmse           = 0.0;
    bool        argmax_match   = true;
    int         argmax_actual  = -1;
    int         argmax_ref     = -1;
    std::size_t nan_count      = 0;
    std::size_t inf_count      = 0;
    bool        all_finite     = true;
};

inline auto check_finite(const std::vector<float>& data) -> std::pair<std::size_t, std::size_t> {
    std::size_t nan_c = 0;
    std::size_t inf_c = 0;
    for (float v : data) {
        if (std::isnan(v)) ++nan_c;
        else if (std::isinf(v)) ++inf_c;
    }
    return {nan_c, inf_c};
}

inline auto compute_error_stats(const std::vector<float>& actual,
                                const std::vector<float>& ref) -> ErrorStats {
    ErrorStats err;
    auto [nan_c, inf_c] = check_finite(actual);
    err.nan_count = nan_c;
    err.inf_count = inf_c;
    err.all_finite = (nan_c == 0 && inf_c == 0);

    if (actual.empty() || ref.empty()) return err;

    const std::size_t n = std::min(actual.size(), ref.size());
    double sum_sq = 0.0;
    float max_d = 0.0f;

    float max_val_act = actual[0];
    int max_idx_act = 0;
    float max_val_ref = ref[0];
    int max_idx_ref = 0;

    for (std::size_t i = 0; i < n; ++i) {
        float a = actual[i];
        float r = ref[i];
        float d = std::abs(a - r);
        if (d > max_d) max_d = d;
        sum_sq += static_cast<double>(d) * static_cast<double>(d);

        if (a > max_val_act) {
            max_val_act = a;
            max_idx_act = static_cast<int>(i);
        }
        if (r > max_val_ref) {
            max_val_ref = r;
            max_idx_ref = static_cast<int>(i);
        }
    }

    err.max_diff = max_d;
    err.rmse = std::sqrt(sum_sq / static_cast<double>(n));
    err.argmax_actual = max_idx_act;
    err.argmax_ref = max_idx_ref;
    err.argmax_match = (max_idx_act == max_idx_ref);
    return err;
}

} // namespace velomind::benchmark
