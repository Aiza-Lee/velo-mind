#include "tinyllama_engine.h"

#include <functional>
#include <memory>
#include <span>
#include <utility>

#include "model_loader.h"
#include "tokenizer.h"
#include "velomind/safetensors.h"

namespace velomind::examples::tinyllama {

using velomind::examples::llama::LlamaEngine;
using velomind::examples::llama::ForwardWeights;

namespace {

auto make_bind_weights()
    -> LlamaEngine::BindWeightsFn {
    return [](velomind::Graph& g, const Config& cfg, std::size_t seq_len)
        -> ForwardWeights {
        return bind_tinyllama_weights(g, cfg, seq_len);
    };
}

auto make_populate_weights()
    -> LlamaEngine::PopulateWeightsFn {
    return [](const velomind::SafetensorsFile& sf, const Config& cfg,
              ForwardWeights& w, std::size_t seq_len) -> void {
        populate_tinyllama_weights(sf, cfg, w, seq_len);
    };
}

auto make_tokenizer_factory() -> LlamaEngine::TokenizerFactory {
    return []() -> std::unique_ptr<velomind::examples::llama::TokenizerAdapter> {

        return std::unique_ptr<velomind::examples::llama::TokenizerAdapter>(
            std::make_unique<Tokenizer>());
    };
}

}

TinyLlamaEngine::TinyLlamaEngine(EngineConfig cfg)
    : cfg_(std::move(cfg)) {

    if (cfg_.use_synthetic) {
        cfg_.model_config = kTiny;
    }
    velomind::examples::llama::LlamaEngineConfig common{
        .model_path     = cfg_.model_path,
        .tokenizer_path = cfg_.tokenizer_path,
        .device         = cfg_.device,
        .use_synthetic  = cfg_.use_synthetic,
        .model_config   = cfg_.model_config,
    };
    engine_.reset(new LlamaEngine(
        std::move(common),
        make_bind_weights(),
        make_populate_weights(),
        LlamaEngine::FillSyntheticFn{},
        make_tokenizer_factory()));
}

void TinyLlamaEngine::load() {
    engine_->load();
}

bool TinyLlamaEngine::is_loaded() const noexcept {
    return engine_ != nullptr && engine_->is_loaded();
}

const velomind::examples::llama::TokenizerAdapter&
TinyLlamaEngine::tokenizer() const noexcept {
    return engine_->tokenizer();
}

const EngineConfig& TinyLlamaEngine::config() const noexcept {
    return cfg_;
}

auto TinyLlamaEngine::prefill(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    return engine_->prefill(tokens);
}

auto TinyLlamaEngine::forward_step(std::int32_t token, std::size_t pos)
    -> std::vector<float> {
    return engine_->forward_step(token, pos);
}

auto TinyLlamaEngine::forward_logits(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    return engine_->forward_logits(tokens);
}

auto TinyLlamaEngine::forward_full_logits(std::span<const std::int32_t> tokens)
    -> std::vector<float> {
    return engine_->forward_full_logits(tokens);
}

auto TinyLlamaEngine::generate(
    const std::string&                                         prompt,
    const std::function<std::int32_t(std::span<const float>)>& sampler_fn,
    std::size_t                                                max_new_tokens,
    std::uint32_t                                              seed,
    std::function<void(std::int32_t, const std::string&)>      on_token)
    -> velomind::examples::llama::LlamaGenerationResult {
    return engine_->generate(prompt, sampler_fn, max_new_tokens, seed,
                             std::move(on_token));
}

auto TinyLlamaEngine::generate(
    std::span<const std::int32_t>                              prompt_tokens,
    const std::function<std::int32_t(std::span<const float>)>& sampler_fn,
    std::size_t                                                max_new_tokens,
    std::uint32_t                                              seed,
    std::function<void(std::int32_t, const std::string&)>      on_token)
    -> velomind::examples::llama::LlamaGenerationResult {
    return engine_->generate(prompt_tokens, sampler_fn, max_new_tokens, seed,
                             std::move(on_token));
}

}
