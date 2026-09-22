#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama_tokenizer.h"

namespace velomind::examples::smollm2 {

class HuggingFaceBpeTokenizer final : public velomind::examples::llama::TokenizerAdapter {
public:
    HuggingFaceBpeTokenizer();
    ~HuggingFaceBpeTokenizer() override = default;

    void load(const std::string& model_path) override;
    auto encode(const std::string& text) const
        -> std::vector<std::int32_t> override;
    auto decode(std::span<const std::int32_t> ids) const
        -> std::string override;

    int  vocab_size() const noexcept override { return static_cast<int>(id_to_token_.size()); }
    int  bos_id()     const noexcept override { return bos_id_; }
    int  eos_id()     const noexcept override { return eos_id_; }
    int  pad_id()     const noexcept override { return pad_id_; }
    int  unk_id()     const noexcept override { return unk_id_; }
    bool loaded()     const noexcept override { return !id_to_token_.empty(); }

    static constexpr std::uint32_t kUnmapped = 0xFFFFFFFFu;

private:

    std::vector<std::string>                       id_to_token_;
    std::unordered_map<std::string, std::int32_t>  token_to_id_;

    int bos_id_ = -1;
    int eos_id_ = -1;
    int pad_id_ = -1;
    int unk_id_ = -1;

    std::vector<std::pair<std::string, std::string>> merges_;

    int byte_to_unicode_[256]   = {0};

    std::vector<std::uint32_t>    unicode_to_byte_;
};

}
