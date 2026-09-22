#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace velomind::examples::llama {

class TokenizerAdapter {
public:
    virtual ~TokenizerAdapter() = default;

    virtual void load(const std::string& model_path) = 0;

    [[nodiscard]] virtual auto encode(const std::string& text) const
        -> std::vector<std::int32_t> = 0;

    [[nodiscard]] virtual auto decode(std::span<const std::int32_t> ids) const
        -> std::string = 0;

    [[nodiscard]] virtual int  vocab_size() const noexcept = 0;
    [[nodiscard]] virtual int  bos_id()     const noexcept = 0;
    [[nodiscard]] virtual int  eos_id()     const noexcept = 0;
    [[nodiscard]] virtual int  pad_id()     const noexcept = 0;
    [[nodiscard]] virtual int  unk_id()     const noexcept = 0;
    [[nodiscard]] virtual bool loaded()     const noexcept = 0;
};

}
