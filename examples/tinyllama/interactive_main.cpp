#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "llama_engine.h"
#include "llama_model_finder.h"
#include "sample.h"
#include "tinyllama_engine.h"
#include "velomind/device.h"

using namespace velomind;
using namespace velomind::examples::tinyllama;
using velomind::examples::llama::run_interactive_console;
using velomind::examples::llama::make_top_p_sampler;

int main(int argc, char** argv) {
    EngineConfig engine_cfg;
    SamplerConfig sampler_cfg{.temperature = 0.7f, .top_p = 0.9f};
    std::size_t max_new_tokens = 32;
    std::string single_prompt;
    bool auto_download = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: tinyllama_chat [options]\n\n"
                      << "Options:\n"
                      << "  --device <cpu|ispc|cuda> Target compute device (default: ispc if available)\n"
                      << "  --model <path>         Path to model.safetensors\n"
                      << "  --tokenizer <path>     Path to tokenizer.model\n"
                      << "  --download             Automatically download model assets if not present\n"
                      << "  --synthetic            Use fast synthetic (kTiny) model for testing\n"
                      << "  --temp <float>         Sampling temperature (default: 0.7, 0 for greedy)\n"
                      << "  --top-p <float>        Nucleus sampling top-p (default: 0.9)\n"
                      << "  --max-tokens <int>     Max generated tokens per turn (default: 32)\n"
                      << "  --prompt <string>      Single non-interactive prompt\n"
                      << "  -h, --help             Show this help message\n";
            return 0;
        } else if (arg == "--device" && i + 1 < argc) {
            std::string dev_str = argv[++i];
            auto parsed_dev = Device::from_string(dev_str);
            if (!parsed_dev) {
                std::cerr << "Unknown device: " << dev_str << " (expected cpu, ispc, or cuda)\n";
                return 1;
            }
            engine_cfg.device = parsed_dev->type();
        } else if (arg == "--model" && i + 1 < argc) {
            engine_cfg.model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            engine_cfg.tokenizer_path = argv[++i];
        } else if (arg == "--download") {
            auto_download = true;
        } else if (arg == "--synthetic") {
            engine_cfg.use_synthetic = true;
        } else if (arg == "--temp" && i + 1 < argc) {
            sampler_cfg.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            sampler_cfg.top_p = std::stof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoul(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            single_prompt = argv[++i];
        }
    }

    try {
        Device dev(engine_cfg.device);

        if (!dev.is_available()) {
            std::cerr << "Error: Target compute device [" << dev.name()
                      << "] is not available or operational on this system.\n";
            if (engine_cfg.device == DeviceType::CUDA) {
                std::cerr << "       (Check GPU drivers / nvidia-smi status, or run with '--device ispc' or '--device cpu')\n";
            }
            return 1;
        }

        if (!engine_cfg.use_synthetic) {
            auto assets = velomind::examples::llama::resolve_model_assets(
                velomind::examples::llama::ModelFamily::TinyLlama,
                engine_cfg.model_path,
                engine_cfg.tokenizer_path
            );
            if (!assets.found) {
                if (!velomind::examples::llama::ensure_model_assets_or_prompt(
                        velomind::examples::llama::ModelFamily::TinyLlama, assets, auto_download)) {
                    velomind::examples::llama::print_missing_model_guide(
                        velomind::examples::llama::ModelFamily::TinyLlama, argv[0]);
                    return 1;
                }
            }
            engine_cfg.model_path = assets.model_path.string();
            engine_cfg.tokenizer_path = assets.tokenizer_path.string();
            std::cout << "Using model assets: " << assets.source_desc << "\n"
                      << "  Directory: " << assets.model_dir.string() << "\n";
        }

        std::cout << "Initializing TinyLlama engine on device ["
                  << dev.name() << "]...\n";
        TinyLlamaEngine engine(engine_cfg);
        engine.load();
        std::cout << "Engine ready.\n";

        if (!single_prompt.empty()) {
            std::cout << "Prompt: " << single_prompt << "\n";
            std::cout << "Output: " << std::flush;
            std::mt19937 rng(42);
            auto sampler_fn = make_top_p_sampler(
                sampler_cfg.temperature, sampler_cfg.top_p, rng);
            auto res = engine.generate(
                single_prompt,
                sampler_fn,
                max_new_tokens,
                42,
                [](std::int32_t , const std::string& piece) {
                    std::cout << piece << std::flush;
                }
            );
            std::cout << "\n\n[" << res.generated_tokens.size() << " tokens in "
                      << res.elapsed_seconds << "s ("
                      << res.tokens_per_second << " tok/s)]\n";
            return 0;
        }

        run_interactive_console(engine.inner(), max_new_tokens, 42);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
