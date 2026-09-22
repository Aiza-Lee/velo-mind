#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "llama_engine.h"
#include "model.h"

namespace velomind::examples::tinyllama {

struct EngineConfig {
    std::string model_path =
        "/home/aiza/workspace/assets/ai-models/TinyLlama_v1.1/model.safetensors";
    std::string tokenizer_path =
        "/home/aiza/workspace/assets/ai-models/TinyLlama_v1.1/tokenizer.model";
#ifdef VELOMIND_ENABLE_ISPC
    velomind::DeviceType device = velomind::DeviceType::ISPC;
#else
    velomind::DeviceType device = velomind::DeviceType::CPU;
#endif
    bool   use_synthetic = false;
    Config model_config  = kReal;
};

class TinyLlamaEngine {
public:
    explicit TinyLlamaEngine(EngineConfig cfg = {});

    void load();

    [[nodiscard]] bool is_loaded() const noexcept;

    [[nodiscard]] const velomind::examples::llama::TokenizerAdapter&
    tokenizer() const noexcept;

    [[nodiscard]] const EngineConfig& config() const noexcept;

    [[nodiscard]] velomind::examples::llama::LlamaEngine& inner() noexcept {
        return *engine_;
    }
    [[nodiscard]] const velomind::examples::llama::LlamaEngine&
    inner() const noexcept { return *engine_; }

    void reset_session() noexcept { engine_->reset_session(); }
    [[nodiscard]] std::size_t session_seq_len() const noexcept {
        return engine_->session_seq_len();
    }

    auto prefill(std::span<const std::int32_t> tokens) -> std::vector<float>;
    auto forward_step(std::int32_t token, std::size_t pos) -> std::vector<float>;

    [[nodiscard]] auto forward_logits(std::span<const std::int32_t> tokens)
        -> std::vector<float>;

    [[nodiscard]] auto forward_full_logits(std::span<const std::int32_t> tokens)
        -> std::vector<float>;

    auto generate(
        const std::string&                                         prompt,
        const std::function<std::int32_t(std::span<const float>)>& sampler_fn,
        std::size_t                                                max_new_tokens,
        std::uint32_t                                              seed = 42,
        std::function<void(std::int32_t, const std::string&)>      on_token = nullptr)
        -> velomind::examples::llama::LlamaGenerationResult;

    auto generate(
        std::span<const std::int32_t>                              prompt_tokens,
        const std::function<std::int32_t(std::span<const float>)>& sampler_fn,
        std::size_t                                                max_new_tokens,
        std::uint32_t                                              seed = 42,
        std::function<void(std::int32_t, const std::string&)>      on_token = nullptr)
        -> velomind::examples::llama::LlamaGenerationResult;

private:
    EngineConfig                                  cfg_;
    std::unique_ptr<velomind::examples::llama::LlamaEngine> engine_;
};

using TextStreamer = velomind::examples::llama::LlamaTextStreamer;
using GenerationResult = velomind::examples::llama::LlamaGenerationResult;

}
