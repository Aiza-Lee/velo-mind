#include "tokenizer.h"

#include <stdexcept>
#include <string>
#include <utility>

#include <sentencepiece_processor.h>

namespace velomind::examples::tinyllama {

Tokenizer::Tokenizer() = default;

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(Tokenizer&& other) noexcept
    : sp_(std::move(other.sp_)),
      vocab_(other.vocab_),
      bos_(other.bos_),
      eos_(other.eos_),
      pad_(other.pad_),
      unk_(other.unk_) {
    other.vocab_ = 0;
    other.bos_   = -1;
    other.eos_   = -1;
    other.pad_   = -1;
    other.unk_   = -1;
}

Tokenizer& Tokenizer::operator=(Tokenizer&& other) noexcept {
    if (this != &other) {
        sp_     = std::move(other.sp_);
        vocab_  = other.vocab_;
        bos_    = other.bos_;
        eos_    = other.eos_;
        pad_    = other.pad_;
        unk_    = other.unk_;
        other.vocab_ = 0;
        other.bos_   = -1;
        other.eos_   = -1;
        other.pad_   = -1;
        other.unk_   = -1;
    }
    return *this;
}

void Tokenizer::load(const std::string& model_path) {
    auto local = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto status = local->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error(
            "tinyllama::Tokenizer: failed to load SentencePiece model '" +
            model_path + "': " + status.ToString());
    }
    sp_     = std::move(local);
    vocab_  = sp_->GetPieceSize();
    bos_    = sp_->bos_id();
    eos_    = sp_->eos_id();
    pad_    = sp_->pad_id();
    unk_    = sp_->unk_id();
}

auto Tokenizer::encode(const std::string& text) const -> std::vector<std::int32_t> {
    if (sp_ == nullptr) {
        throw std::runtime_error(
            "tinyllama::Tokenizer::encode: model not loaded");
    }
    std::vector<int> ids;
    auto status = sp_->Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error(
            "tinyllama::Tokenizer::encode: " + status.ToString());
    }
    return std::vector<std::int32_t>(ids.begin(), ids.end());
}

auto Tokenizer::decode(std::span<const std::int32_t> ids) const -> std::string {
    if (sp_ == nullptr) {
        throw std::runtime_error(
            "tinyllama::Tokenizer::decode: model not loaded");
    }
    std::vector<int> sp_ids(ids.begin(), ids.end());
    std::string text;
    auto status = sp_->Decode(sp_ids, &text);
    if (!status.ok()) {
        throw std::runtime_error(
            "tinyllama::Tokenizer::decode: " + status.ToString());
    }
    return text;
}

int  Tokenizer::vocab_size() const noexcept { return vocab_; }
int  Tokenizer::bos_id()     const noexcept { return bos_;   }
int  Tokenizer::eos_id()     const noexcept { return eos_;   }
int  Tokenizer::pad_id()     const noexcept { return pad_;   }
int  Tokenizer::unk_id()     const noexcept { return unk_;   }
bool Tokenizer::loaded()     const noexcept { return sp_ != nullptr; }

}
