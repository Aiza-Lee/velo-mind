#include "sample.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "velomind/types.h"

namespace velomind::examples::tinyllama {

auto sample_argmax(std::span<const float> logits) -> std::int32_t {
    if (logits.empty()) {
        throw std::invalid_argument("sample_argmax: logits span is empty");
    }
    std::int32_t best_idx = -1;
    float best_val = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
        float v = logits[i];
        if (std::isnan(v)) {
            continue;
        }
        if (best_idx == -1 || v > best_val) {
            best_val = v;
            best_idx = static_cast<std::int32_t>(i);
        }
    }
    if (best_idx == -1) {
        throw std::runtime_error("sample_argmax: all logits are NaN");
    }
    return best_idx;
}

auto sample_top_p(std::span<const float> logits,
                  float                  temperature,
                  float                  top_p,
                  std::mt19937&          rng) -> std::int32_t {
    if (logits.empty()) {
        throw std::invalid_argument("sample_top_p: logits span is empty");
    }
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return sample_argmax(logits);
    }

    float max_val = -std::numeric_limits<float>::infinity();
    bool has_finite = false;
    for (float v : logits) {
        if (std::isfinite(v)) {
            if (!has_finite || v > max_val) {
                max_val = v;
                has_finite = true;
            }
        }
    }
    if (!has_finite) {
        return sample_argmax(logits);
    }

    const float inv_t = 1.0f / temperature;
    struct TokenProb {
        float        prob;
        std::int32_t id;
    };
    std::vector<TokenProb> candidates;
    candidates.reserve(logits.size());

    double sum_exp = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        float v = logits[i];
        if (std::isnan(v)) {
            continue;
        }
        float p = std::exp((v - max_val) * inv_t);
        sum_exp += p;
        candidates.push_back({p, static_cast<std::int32_t>(i)});
    }

    if (candidates.empty()) {
        throw std::runtime_error("sample_top_p: no valid candidate tokens");
    }

    if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
        return sample_argmax(logits);
    }

    const float inv_sum = static_cast<float>(1.0 / sum_exp);
    for (auto& c : candidates) {
        c.prob *= inv_sum;
    }

    std::sort(candidates.begin(), candidates.end(), [](const TokenProb& a, const TokenProb& b) {
        return a.prob > b.prob;
    });

    float cum_prob = 0.0f;
    std::size_t cutoff = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        cum_prob += candidates[i].prob;
        cutoff = i + 1;
        if (cum_prob >= top_p) {
            break;
        }
    }
    if (cutoff == 0 && !candidates.empty()) {
        cutoff = 1;
        cum_prob = candidates[0].prob;
    }

    std::uniform_real_distribution<float> dist(0.0f, cum_prob);
    float r = dist(rng);
    float acc = 0.0f;
    for (std::size_t i = 0; i < cutoff; ++i) {
        acc += candidates[i].prob;
        if (r <= acc) {
            return candidates[i].id;
        }
    }
    return candidates[cutoff - 1].id;
}

auto sample_token(std::span<const float> logits,
                  const SamplerConfig&   config,
                  std::mt19937&          rng) -> std::int32_t {
    if (config.temperature <= 0.0f || config.top_p <= 0.0f) {
        return sample_argmax(logits);
    }
    return sample_top_p(logits, config.temperature, config.top_p, rng);
}

auto read_last_token_logits(const Tensor& logits) -> std::vector<float> {
    if (!logits) {
        throw std::runtime_error("read_last_token_logits: null Tensor handle");
    }
    if (logits.dtype() != DataType::Float32) {
        throw std::runtime_error("read_last_token_logits: expected Float32 tensor");
    }
    const auto& s = logits.shape();
    if (s.empty()) {
        throw std::runtime_error("read_last_token_logits: empty tensor shape");
    }
    std::size_t vocab_size = 0;
    std::size_t offset     = 0;
    if (s.size() == 1) {
        vocab_size = static_cast<std::size_t>(s[0]);
        offset     = 0;
    } else if (s.size() == 2) {
        std::size_t seq = static_cast<std::size_t>(s[0]);
        vocab_size      = static_cast<std::size_t>(s[1]);
        if (seq == 0 || vocab_size == 0) {
            throw std::runtime_error("read_last_token_logits: zero seq or vocab size");
        }
        offset = (seq - 1) * vocab_size;
    } else {
        throw std::runtime_error(
            "read_last_token_logits: tensor must be 1-D [vocab] or 2-D [seq, vocab]");
    }

    std::vector<float> full_buf(logits.numel());
    auto byte_span = std::span<std::byte>(
        reinterpret_cast<std::byte*>(full_buf.data()), full_buf.size() * sizeof(float));
    logits.copy_to_host(byte_span);

    std::vector<float> last_logits(vocab_size);
    std::copy_n(full_buf.begin() + offset, vocab_size, last_logits.begin());
    return last_logits;
}

auto sample_next_token(const Tensor&         logits,
                       const SamplerConfig&  config,
                       std::mt19937&         rng) -> std::int32_t {
    auto last_logits = read_last_token_logits(logits);
    return sample_token(last_logits, config, rng);
}

auto generate_tokens(
    const std::vector<std::int32_t>& prompt,
    std::size_t                      max_new_tokens,
    const SamplerConfig&             config,
    std::int32_t                     eos_token_id,
    std::mt19937&                    rng,
    const std::function<std::vector<float>(std::span<const std::int32_t>)>& forward_fn,
    const std::function<void(std::int32_t)>& on_token) -> std::vector<std::int32_t> {
    std::vector<std::int32_t> current = prompt;
    std::vector<std::int32_t> generated;
    generated.reserve(max_new_tokens);

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        auto logits = forward_fn(current);
        if (logits.empty()) {
            break;
        }
        std::int32_t next_token = sample_token(logits, config, rng);
        generated.push_back(next_token);
        current.push_back(next_token);
        if (on_token) {
            on_token(next_token);
        }
        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break;
        }
    }
    return generated;
}

auto generate_tokens(
    const std::vector<std::int32_t>& prompt,
    std::size_t                      max_new_tokens,
    const SamplerConfig&             config,
    std::int32_t                     eos_token_id,
    std::mt19937&                    rng,
    const std::function<Tensor(std::span<const std::int32_t>)>& forward_tensor_fn,
    const std::function<void(std::int32_t)>& on_token) -> std::vector<std::int32_t> {
    return generate_tokens(
        prompt, max_new_tokens, config, eos_token_id, rng,
        [&](std::span<const std::int32_t> tokens) {
            Tensor t = forward_tensor_fn(tokens);
            return read_last_token_logits(t);
        },
        on_token);
}

auto make_sampler_fn(const SamplerConfig& config, std::mt19937& rng)
    -> std::function<std::int32_t(std::span<const float>)> {
    return [config, &rng](std::span<const float> logits) -> std::int32_t {
        return sample_token(logits, config, rng);
    };
}

}
