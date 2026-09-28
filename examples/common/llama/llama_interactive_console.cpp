#include "llama_interactive_console.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "velomind/device.h"
#include "llama_engine.h"

namespace velomind::examples::llama {

namespace {

struct ConsoleState {
    float         temperature    = 0.7f;
    float         top_p          = 0.9f;
    std::size_t   max_new_tokens = 64;
    std::uint32_t seed           = 42;

    std::string system_prompt;

    std::vector<std::string> turns;
    bool                     have_system = false;

    std::size_t last_tokens  = 0;
    double      last_seconds = 0.0;
    double      last_tps     = 0.0;
    std::string last_text;
};

auto make_default_sampler(const ConsoleState& state)
    -> std::function<std::int32_t(std::span<const float>)> {
    return [state](std::span<const float> logits) -> std::int32_t {
        if (logits.empty()) return -1;
        if (state.temperature <= 0.0f || state.top_p <= 0.0f) {
            std::int32_t best = -1;
            float bv = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < logits.size(); ++i) {
                float v = logits[i];
                if (std::isnan(v)) continue;
                if (best == -1 || v > bv) { bv = v; best = static_cast<std::int32_t>(i); }
            }
            return best;
        }

        float max_v = -std::numeric_limits<float>::infinity();
        bool any = false;
        for (float v : logits) {
            if (std::isfinite(v) && (!any || v > max_v)) { max_v = v; any = true; }
        }
        if (!any) return -1;
        const float inv_t = 1.0f / state.temperature;
        struct Prob { float p; std::int32_t id; };
        std::vector<Prob> cands;
        cands.reserve(logits.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            float v = logits[i];
            if (std::isnan(v)) continue;
            float p = std::exp((v - max_v) * inv_t);
            sum += p;
            cands.push_back({p, static_cast<std::int32_t>(i)});
        }
        if (cands.empty()) return -1;
        const float inv_sum = static_cast<float>(1.0 / sum);
        for (auto& c : cands) c.p *= inv_sum;
        std::sort(cands.begin(), cands.end(),
                  [](const Prob& a, const Prob& b) { return a.p > b.p; });
        float cum = 0.0f;
        std::size_t cutoff = 0;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].p;
            cutoff = i + 1;
            if (cum >= state.top_p) break;
        }
        if (cutoff == 0 && !cands.empty()) {
            cutoff = 1;
            cum = cands[0].p;
        }
        std::mt19937 rng(static_cast<std::uint32_t>(std::random_device{}()));
        std::uniform_real_distribution<float> dist(0.0f, cum);
        float r = dist(rng);
        float acc = 0.0f;
        for (std::size_t i = 0; i < cutoff; ++i) {
            acc += cands[i].p;
            if (r <= acc) return cands[i].id;
        }
        return cands[cutoff - 1].id;
    };
}

auto compose_prompt(const ConsoleState& state, const std::string& user_text)
    -> std::string {
    std::string out;
    if (state.have_system && !state.system_prompt.empty()) {
        out += state.system_prompt;
        out += "\n";
    }
    for (std::size_t i = 0; i < state.turns.size(); ++i) {
        out += state.turns[i];
        out += "\n";
    }
    out += user_text;
    return out;
}

auto trim(std::string s) -> std::string {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

void print_help(const ConsoleState& state, const LlamaEngine& engine) {
    const auto& cfg = engine.config();
    std::cout << "Commands:\n"
              << "  /help               Show commands + current state.\n"
              << "  /temp <float>       Sampling temperature (0 = greedy). Current: " << state.temperature << "\n"
              << "  /top_p <float>      Nucleus top-p threshold. Current: " << state.top_p << "\n"
              << "  /max <int>          Max generated tokens per response. Current: " << state.max_new_tokens << "\n"
              << "  /system <text>      Set/replace the system prompt prefix.\n"
              << "  /clear              Clear conversation history (keeps system).\n"
              << "  /history            Show per-turn stats for the last response.\n"
              << "  /model              Show model info (path / config / device).\n"
              << "  /save <file>        Persist conversation history to file.\n"
              << "  /load <file>        Load conversation history from file.\n"
              << "  /exit, /quit        Exit.\n\n"
              << "Model path: "    << cfg.model_path     << "\n"
              << "Tokenizer:  "    << cfg.tokenizer_path << "\n"
              << "Device:     "    << Device(cfg.device).name() << "\n"
              << "Synthetic:  "    << (cfg.use_synthetic ? "yes" : "no") << "\n"
              << "Vocab:      "    << engine.tokenizer().vocab_size() << " tokens\n";
}

void print_model_info(const LlamaEngine& engine) {
    const auto& cfg = engine.config().model_config;
    std::cout << "Backend:  " << Device(engine.config().device).name() << "\n"
              << "Mode:     " << (engine.config().use_synthetic ? "Synthetic" : "Real model") << "\n"
              << "Vocab:    " << engine.tokenizer().vocab_size() << " tokens\n"
              << "Hidden:   " << cfg.hidden_size      << "\n"
              << "Layers:   " << cfg.num_layers        << "\n"
              << "Heads:    " << cfg.num_heads         << " (kv=" << cfg.num_kv_heads << ")\n"
              << "HeadDim:  " << cfg.head_dim()        << "\n"
              << "Intermed: " << cfg.intermediate_size << "\n"
              << "Rope θ:   " << cfg.rope_theta        << "\n"
              << "Tied:     " << (cfg.tie_word_embeddings ? "yes" : "no") << "\n";
}

auto save_conversation(const ConsoleState& state, const std::string& path) -> bool {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[save] cannot open " << path << " for writing\n";
        return false;
    }
    f << "# aito conversation snapshot\n";
    if (state.have_system) {
        f << "SYSTEM\n" << state.system_prompt << "\n";
    }
    for (std::size_t i = 0; i < state.turns.size(); ++i) {
        f << ((i % 2 == 0) ? "USER\n" : "ASSISTANT\n")
          << state.turns[i] << "\n";
    }
    return f.good();
}

auto load_conversation(ConsoleState& state, const std::string& path) -> bool {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[load] cannot open " << path << " for reading\n";
        return false;
    }
    state.turns.clear();
    state.have_system = false;
    state.system_prompt.clear();
    std::string line, role, body;
    auto flush = [&]() {
        if (role.empty()) return;
        if (role == "SYSTEM") {
            state.have_system = true;
            state.system_prompt = body;
        } else if (role == "USER" || role == "ASSISTANT") {
            state.turns.push_back(body);
        }
        role.clear();
        body.clear();
    };
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') {
            flush();
            continue;
        }
        if (line == "SYSTEM" || line == "USER" || line == "ASSISTANT") {
            flush();
            role = line;
        } else {
            if (!body.empty()) body += "\n";
            body += line;
        }
    }
    flush();

    return !f.bad();
}

// 交互控制台指令处理状态
enum class ConsoleAction {
    Handled,
    Exit,
    Passthrough
};

// 处理控制台斜杠指令
auto dispatch_console_command(
    const std::string& cmd,
    ConsoleState&      state,
    LlamaEngine&       engine
) -> ConsoleAction {
    if (cmd == "/exit" || cmd == "/quit") {
        std::cout << "Exiting...\n";
        return ConsoleAction::Exit;
    }
    if (cmd == "/help") {
        print_help(state, engine);
        std::cout << "\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/model") {
        print_model_info(engine);
        std::cout << "\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/history") {
        std::cout << "Last response: " << state.last_tokens << " tokens in "
                  << std::fixed << std::setprecision(2) << state.last_seconds
                  << "s (" << std::setprecision(1) << state.last_tps << " tok/s)\n"
                  << "Last text: " << state.last_text << "\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/clear") {
        state.turns.clear();
        std::cout << "Conversation cleared (system prompt retained).\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/temp ", 0) == 0) {
        try {
            state.temperature = std::stof(cmd.substr(6));
            std::cout << "Temperature set to " << state.temperature << "\n\n";
        } catch (...) { std::cout << "Invalid temperature value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/top_p ", 0) == 0) {
        try {
            state.top_p = std::stof(cmd.substr(7));
            std::cout << "Top-P set to " << state.top_p << "\n\n";
        } catch (...) { std::cout << "Invalid top-p value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/max ", 0) == 0) {
        try {
            state.max_new_tokens = static_cast<std::size_t>(std::stoul(cmd.substr(5)));
            std::cout << "Max new tokens set to " << state.max_new_tokens << "\n\n";
        } catch (...) { std::cout << "Invalid max tokens value\n\n"; }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/system ", 0) == 0) {
        state.system_prompt = cmd.substr(8);
        state.have_system = !state.system_prompt.empty();
        std::cout << "System prompt set (" << state.system_prompt.size()
                  << " chars).\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd == "/system") {
        state.system_prompt.clear();
        state.have_system = false;
        std::cout << "System prompt cleared.\n\n";
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/save ", 0) == 0) {
        std::string path = trim(cmd.substr(6));
        if (save_conversation(state, path)) {
            std::cout << "Saved to " << path << "\n\n";
        } else {
            std::cout << "Save failed.\n\n";
        }
        return ConsoleAction::Handled;
    }
    if (cmd.rfind("/load ", 0) == 0) {
        std::string path = trim(cmd.substr(6));
        if (load_conversation(state, path)) {
            std::cout << "Loaded from " << path << " ("
                      << state.turns.size() << " turns)\n\n";
        } else {
            std::cout << "Load failed.\n\n";
        }
        return ConsoleAction::Handled;
    }
    return ConsoleAction::Passthrough;
}

// 执行单轮交互生成并记录历史
auto execute_generation_turn(
    LlamaEngine&       engine,
    ConsoleState&      state,
    const std::string& cmd
) -> void {
    try {
        std::string composed = compose_prompt(state, cmd);
        auto sampler = make_default_sampler(state);
        auto res = engine.generate(
            composed,
            sampler,
            state.max_new_tokens,
            state.seed++,
            [](std::int32_t, const std::string& piece) {
                std::cout << piece << std::flush;
            }
        );
        std::cout << "\n";
        std::cout << "[" << res.generated_tokens.size() << " tokens in "
                  << std::fixed << std::setprecision(2) << res.elapsed_seconds
                  << "s (" << std::setprecision(1) << res.tokens_per_second
                  << " tok/s)]\n\n";

        state.last_tokens  = res.generated_tokens.size();
        state.last_seconds = res.elapsed_seconds;
        state.last_tps     = res.tokens_per_second;
        state.last_text    = res.text;

        state.turns.push_back(cmd);
        state.turns.push_back(res.text);
    } catch (const std::exception& ex) {
        std::cerr << "\n[Error during generation: " << ex.what() << "]\n\n";
    }
}

} // namespace

void run_interactive_console(
    LlamaEngine&  engine,
    std::size_t   max_new_tokens,
    std::uint32_t seed
) {
    ConsoleState state;
    state.max_new_tokens = max_new_tokens;
    state.seed           = seed;

    const auto& dev_cfg = engine.config().device;
    std::cout << "\n"
              << "===================================================================\n"
              << " Backend:  " << Device(dev_cfg).name() << "\n"
              << " Mode:     " << (engine.config().use_synthetic ? "Synthetic" : "Full Model") << "\n"
              << " Vocab:    " << engine.tokenizer().vocab_size() << " tokens\n"
              << " Commands: /help /temp /top_p /max /system /clear /history\n"
              << "           /model /save /load /exit\n"
              << "===================================================================\n\n";

    std::string line;
    while (true) {
        std::cout << ">>> " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\nExiting...\n";
            break;
        }
        std::string cmd = trim(line);
        if (cmd.empty()) continue;

        const auto action = dispatch_console_command(cmd, state, engine);
        if (action == ConsoleAction::Exit) {
            break;
        }
        if (action == ConsoleAction::Handled) {
            continue;
        }

        execute_generation_turn(engine, state, cmd);
    }
}

} // namespace velomind::examples::llama
