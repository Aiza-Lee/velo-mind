#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

#include "velomind/tensor.h"

namespace velomind::examples::tinyllama {

struct SamplerConfig {

    float temperature = 1.0f;

    float top_p = 0.9f;
};

auto sample_argmax(std::span<const float> logits) -> std::int32_t;

auto sample_top_p(std::span<const float> logits,
                  float                  temperature,
                  float                  top_p,
                  std::mt19937&          rng) -> std::int32_t;

auto sample_token(std::span<const float> logits,
                  const SamplerConfig&   config,
                  std::mt19937&          rng) -> std::int32_t;

auto read_last_token_logits(const Tensor& logits) -> std::vector<float>;

auto sample_next_token(const Tensor&         logits,
                       const SamplerConfig&  config,
                       std::mt19937&         rng) -> std::int32_t;

auto generate_tokens(
    const std::vector<std::int32_t>& prompt,
    std::size_t                      max_new_tokens,
    const SamplerConfig&             config,
    std::int32_t                     eos_token_id,
    std::mt19937&                    rng,
    const std::function<std::vector<float>(std::span<const std::int32_t>)>& forward_fn,
    const std::function<void(std::int32_t)>& on_token = nullptr) -> std::vector<std::int32_t>;

auto make_sampler_fn(const SamplerConfig& config, std::mt19937& rng)
    -> std::function<std::int32_t(std::span<const float>)>;

}
