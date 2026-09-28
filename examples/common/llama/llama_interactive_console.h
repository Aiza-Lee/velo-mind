#pragma once

#include <cstddef>
#include <cstdint>

namespace velomind::examples::llama {

class LlamaEngine;

// 运行 LLaMA/SmolLM2 交互式命令行控制台（REPL）。
// 支持实时流式输出、参数动态调节（/temp, /top_p, /max）、多轮对话历史持久化（/save, /load）及系统提示词配置。
void run_interactive_console(
    LlamaEngine&  engine,
    std::size_t   max_new_tokens = 64,
    std::uint32_t seed           = 42);

} // namespace velomind::examples::llama
