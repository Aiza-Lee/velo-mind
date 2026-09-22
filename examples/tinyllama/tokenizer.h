#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "llama_tokenizer.h"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace velomind::examples::tinyllama {

class Tokenizer : public velomind::examples::llama::TokenizerAdapter {
public:
    Tokenizer();
    ~Tokenizer() override;
    Tokenizer(const Tokenizer&)            = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    void load(const std::string& model_path) override;

    [[nodiscard]] auto encode(const std::string& text) const
        -> std::vector<std::int32_t> override;

    [[nodiscard]] auto decode(std::span<const std::int32_t> ids) const
        -> std::string override;

    [[nodiscard]] int  vocab_size() const noexcept override;
    [[nodiscard]] int  bos_id()     const noexcept override;
    [[nodiscard]] int  eos_id()     const noexcept override;
    [[nodiscard]] int  pad_id()     const noexcept override;
    [[nodiscard]] int  unk_id()     const noexcept override;
    [[nodiscard]] bool loaded()     const noexcept override;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    int vocab_ = 0;
    int bos_   = -1;
    int eos_   = -1;
    int pad_   = -1;
    int unk_   = -1;
};

}
