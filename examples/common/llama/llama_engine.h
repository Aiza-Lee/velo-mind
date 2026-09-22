#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "velomind/safetensors.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include "llama_config.h"
#include "llama_graph.h"
#include "llama_tokenizer.h"

namespace velomind::examples::llama {

struct LlamaEngineConfig {
    std::string model_path;
    std::string tokenizer_path;
    DeviceType  device        = DeviceType::CPU;
    bool        use_synthetic = false;
    Config      model_config  = {};
};

struct LlamaGenerationResult {
    std::vector<std::int32_t> generated_tokens;
    std::string               text;
    std::size_t               prompt_token_count = 0;
    double                    elapsed_seconds    = 0.0;
    double                    tokens_per_second  = 0.0;
};

struct LayerKVCache {
    std::shared_ptr<TensorStorage> k;
    std::shared_ptr<TensorStorage> v;
};

struct SessionKVCache {
    std::vector<LayerKVCache> layers;
    std::size_t               seq_len = 0;

    void clear() noexcept {
        layers.clear();
        seq_len = 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return seq_len == 0;
    }
};

class LlamaEngine {
public:
    using BindWeightsFn = std::function<ForwardWeights(
        Graph&, const Config&, std::size_t seq_len)>;
    using PopulateWeightsFn = std::function<void(
        const SafetensorsFile&, const Config&, ForwardWeights&, std::size_t seq_len)>;
    using FillSyntheticFn = std::function<void(
        Graph&, ForwardWeights&, const Config&)>;
    using TokenizerFactory = std::function<std::unique_ptr<TokenizerAdapter>()>;

    explicit LlamaEngine(LlamaEngineConfig  config,
                         BindWeightsFn      bind_weights,
                         PopulateWeightsFn  populate_weights,
                         FillSyntheticFn    fill_synthetic,
                         TokenizerFactory   tokenizer_factory);
    ~LlamaEngine();

    void load();

    [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

    [[nodiscard]] const TokenizerAdapter& tokenizer() const noexcept {
        return *tokenizer_;
    }

    [[nodiscard]] const LlamaEngineConfig& config() const noexcept {
        return config_;
    }

    void reset_session() noexcept;

    [[nodiscard]] std::size_t session_seq_len() const noexcept;

    [[nodiscard]] const SessionKVCache& kv_cache() const noexcept;

    // 增量 prefill：对输入 token 序列执行前向推理，构建各层 KV cache 并返回末尾 token logits。
    auto prefill(std::span<const std::int32_t> tokens) -> std::vector<float>;

    // 增量 decode：基于当前会话 KV cache 执行单 token 解码，更新各层 cache 并返回下一 token logits。
    auto forward_step(std::int32_t token, std::size_t pos) -> std::vector<float>;

    // 无状态全序列 logits 前向推理，保留用于无 cache 对齐与参考基准校验。
    [[nodiscard]] auto forward_logits(std::span<const std::int32_t> tokens)
        -> std::vector<float>;

    // 无状态全序列 logits 前向推理，输出 [S, vocab] 完整矩阵，用于困惑度评估与参考基准校验。
    [[nodiscard]] auto forward_full_logits(std::span<const std::int32_t> tokens)
        -> std::vector<float>;

    auto generate(
        const std::string&                            prompt,
        const std::function<std::int32_t(
            std::span<const float>)>&                 sample_fn,
        std::size_t                                   max_new_tokens,
        std::uint32_t                                 seed,
        std::function<void(std::int32_t, const std::string&)> on_token = nullptr)
        -> LlamaGenerationResult;

    auto generate(
        std::span<const std::int32_t>                 prompt_tokens,
        const std::function<std::int32_t(
            std::span<const float>)>&                 sample_fn,
        std::size_t                                   max_new_tokens,
        std::uint32_t                                 seed,
        std::function<void(std::int32_t, const std::string&)> on_token = nullptr)
        -> LlamaGenerationResult;

    // 绑定已加载的权重存储到目标计算图，复用同一组物理权重。
    void bind_model_weights(Graph& g, ForwardWeights& step_w) const;

private:
    LlamaEngineConfig                  config_;
    BindWeightsFn                      bind_weights_;
    PopulateWeightsFn                  populate_weights_;
    FillSyntheticFn                    fill_synthetic_;
    TokenizerFactory                   tokenizer_factory_;
    std::unique_ptr<TokenizerAdapter>  tokenizer_;
    bool                               loaded_ = false;

    void load_synthetic_weights_();
    void load_safetensors_weights_();
    void record_weight_storages_(const ForwardWeights& w, std::size_t num_layers);

    SessionKVCache                              session_kv_cache_;
    std::shared_ptr<TensorStorage>              w_embed_storage_;
    std::shared_ptr<TensorStorage>              w_lm_head_storage_;
    std::shared_ptr<TensorStorage>              w_final_norm_storage_;
    std::vector<std::shared_ptr<TensorStorage>> w_attn_norm_storage_;
    std::vector<std::shared_ptr<TensorStorage>> w_ffn_norm_storage_;
    std::vector<std::shared_ptr<TensorStorage>> wq_storage_;
    std::vector<std::shared_ptr<TensorStorage>> wk_storage_;
    std::vector<std::shared_ptr<TensorStorage>> wv_storage_;
    std::vector<std::shared_ptr<TensorStorage>> wo_storage_;
    std::vector<std::shared_ptr<TensorStorage>> w_gate_storage_;
    std::vector<std::shared_ptr<TensorStorage>> w_up_storage_;
    std::vector<std::shared_ptr<TensorStorage>> w_down_storage_;
};

class LlamaTextStreamer {
public:
    explicit LlamaTextStreamer(
        const TokenizerAdapter&                    tokenizer,
        std::function<void(const std::string&)>    callback = nullptr);

    auto put(std::int32_t token_id) -> std::string;

    auto flush() -> std::string;

    void reset(std::span<const std::int32_t> initial_tokens = {});

    [[nodiscard]] auto generated_tokens() const
        -> const std::vector<std::int32_t>& { return generated_tokens_; }

    [[nodiscard]] auto text() const -> const std::string& {
        return full_generated_text_;
    }

private:
    const TokenizerAdapter&              tokenizer_;
    std::function<void(const std::string&)> callback_;
    std::vector<std::int32_t>             all_tokens_;
    std::vector<std::int32_t>             generated_tokens_;
    std::string                           decoded_all_;
    std::string                           full_generated_text_;
    std::size_t                           prompt_tokens_count_ = 0;
};

void run_interactive_console(
    LlamaEngine&      engine,
    std::size_t       max_new_tokens = 64,
    std::uint32_t     seed           = 42);

auto make_top_p_sampler(float temperature, float top_p, std::mt19937& rng)
    -> std::function<std::int32_t(std::span<const float>)>;

}
