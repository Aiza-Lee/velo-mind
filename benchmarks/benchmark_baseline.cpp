#include <format>
#include <iostream>
#include <sstream>
#include <string>

#include "benchmark_runner.h"
#include "llama_model_finder.h"

int main(int argc, char** argv) {
    using namespace velomind;
    using namespace velomind::benchmark;

    BenchmarkRunConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            std::string d = argv[++i];
            if (d == "cpu")         cfg.devices = {DeviceType::CPU};
            else if (d == "ispc")   cfg.devices = {DeviceType::ISPC};
            else if (d == "cuda")   cfg.devices = {DeviceType::CUDA};
            else if (d == "all")    cfg.devices = {DeviceType::CPU, DeviceType::ISPC, DeviceType::CUDA};
        } else if (arg == "--seq-lens" && i + 1 < argc) {
            cfg.seq_lens.clear();
            std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) cfg.seq_lens.push_back(std::stoul(item));
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iterations = std::stoul(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.threads = std::stoul(argv[++i]);
        } else if (arg == "--model-path" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--tokenizer-path" && i + 1 < argc) {
            cfg.tokenizer_path = argv[++i];
        } else if (arg == "--synthetic") {
            cfg.use_synthetic = true;
        } else if (arg == "--markdown" && i + 1 < argc) {
            cfg.markdown_out = argv[++i];
        } else if (arg == "--quiet") {
            cfg.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << std::format(
                "用法: {} [选项]\n"
                "  --device <cpu|ispc|cuda|all>        目标计算设备 (默认: 全部可用设备)\n"
                "  --seq-lens <s1,s2,...>              测试序列长度列表 (例如: 1,32,128,512,2048)\n"
                "  --iters <N>                         测量迭代轮数 (默认: 5)\n"
                "  --warmup <W>                        预热执行轮数 (默认: 1)\n"
                "  --threads <T>                       ISPC 与 CPU 后端的执行线程预算 (默认: 8)\n"
                "  --model-path <path>                 模型权重文件路径 (model.safetensors)\n"
                "  --tokenizer-path <path>             分词器配置文件路径 (tokenizer.json)\n"
                "  --synthetic                         使用随机合成权重进行纯结构评测\n"
                "  --markdown <file>                   导出 Markdown 格式评测矩阵报告\n"
                "  --quiet                             静默模式，仅输出最小化日志\n"
                "  -h, --help                          显示此帮助信息并退出\n",
                argv[0]
            );
            return 0;
        }
    }

    if (!cfg.use_synthetic && (cfg.model_path.empty() || cfg.tokenizer_path.empty())) {
        auto assets = velomind::examples::llama::resolve_model_assets(
            velomind::examples::llama::ModelFamily::SmolLM2,
            cfg.model_path,
            cfg.tokenizer_path
        );
        if (assets.found) {
            cfg.model_path = assets.model_path.string();
            cfg.tokenizer_path = assets.tokenizer_path.string();
        }
    }

    try {
        BenchmarkRunner runner(cfg);
        runner.run();
    } catch (const std::exception& e) {
        std::cerr << std::format("基准评测异常终止: {}\n", e.what());
        return 1;
    }
    return 0;
}
