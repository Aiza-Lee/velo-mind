#include "llama_model_finder.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#else
#include <unistd.h>
#endif

namespace velomind::examples::llama {

namespace fs = std::filesystem;

namespace {

auto is_repo_root_dir(const fs::path& p) -> bool {
    std::error_code ec;
    return fs::exists(p / "CMakeLists.txt", ec) &&
           (fs::exists(p / "include" / "velomind.h", ec) ||
            fs::exists(p / "include" / "velomind", ec));
}

} // namespace

auto find_repo_root() -> fs::path {
#ifdef VELOMIND_PROJECT_ROOT
    const std::string root_macro = VELOMIND_PROJECT_ROOT;
    if (!root_macro.empty()) {
        fs::path p(root_macro);
        std::error_code ec;
        if (fs::exists(p, ec) && is_repo_root_dir(p)) {
            return p;
        }
    }
#endif

#if defined(__linux__)
    std::error_code ec;
    auto exe_symlink = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        auto cur = exe_symlink.parent_path();
        for (int i = 0; i < 6 && !cur.empty(); ++i) {
            if (is_repo_root_dir(cur)) {
                return cur;
            }
            cur = cur.parent_path();
        }
    }
#endif

    auto cur = fs::current_path(ec);
    if (!ec) {
        for (int i = 0; i < 6 && !cur.empty(); ++i) {
            if (is_repo_root_dir(cur)) {
                return cur;
            }
            cur = cur.parent_path();
        }
    }

    return fs::current_path();
}

auto get_user_cache_dir() -> fs::path {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg != '\0') {
            return fs::path(xdg) / "velomind" / "models";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home != '\0') {
            return fs::path(home) / ".cache" / "velomind" / "models";
        }
    }
    return fs::current_path() / ".cache" / "velomind" / "models";
}

auto resolve_model_assets(
    ModelFamily family,
    const std::string& explicit_model_path,
    const std::string& explicit_tokenizer_path
) -> ModelAssetPaths {
    ModelAssetPaths res;
    std::error_code ec;

    const std::string dir_name = (family == ModelFamily::SmolLM2) ? "SmolLM2-135M" : "TinyLlama_v1.1";
    const std::string tok_name = (family == ModelFamily::SmolLM2) ? "tokenizer.json" : "tokenizer.model";

    // 显式传入参数时优先采用
    if (!explicit_model_path.empty()) {
        fs::path mp(explicit_model_path);
        if (fs::exists(mp, ec)) {
            res.model_path = fs::canonical(mp, ec);
            res.model_dir = res.model_path.parent_path();

            if (!explicit_tokenizer_path.empty()) {
                fs::path tp(explicit_tokenizer_path);
                if (fs::exists(tp, ec)) {
                    res.tokenizer_path = fs::canonical(tp, ec);
                }
            } else {
                fs::path inferred_tok = res.model_dir / tok_name;
                if (fs::exists(inferred_tok, ec)) {
                    res.tokenizer_path = fs::canonical(inferred_tok, ec);
                }
            }

            fs::path cfg_p = res.model_dir / "config.json";
            if (fs::exists(cfg_p, ec)) {
                res.config_path = fs::canonical(cfg_p, ec);
            }

            if (!res.tokenizer_path.empty()) {
                res.found = true;
                res.source_desc = "命令行参数显式指定";
                return res;
            }
        } else {
            res.found = false;
            res.source_desc = "指定模型文件不存在: " + explicit_model_path;
            return res;
        }
    }

    // 级联探测候选目录列表
    struct CandidateEntry {
        fs::path    path;
        std::string desc;
    };
    std::vector<CandidateEntry> candidates;

    // 1. 环境变量
    if (family == ModelFamily::SmolLM2) {
        if (const char* env = std::getenv("VELOMIND_SMOLLM2_DIR")) {
            candidates.push_back({fs::path(env), "环境变量 VELOMIND_SMOLLM2_DIR"});
        }
        if (const char* env = std::getenv("VELOMIND_SMOLLM2_PATH")) {
            fs::path p(env);
            candidates.push_back({fs::is_directory(p, ec) ? p : p.parent_path(), "环境变量 VELOMIND_SMOLLM2_PATH"});
        }
    } else {
        if (const char* env = std::getenv("VELOMIND_TINYLLAMA_DIR")) {
            candidates.push_back({fs::path(env), "环境变量 VELOMIND_TINYLLAMA_DIR"});
        }
        if (const char* env = std::getenv("VELOMIND_TINYLLAMA_PATH")) {
            fs::path p(env);
            candidates.push_back({fs::is_directory(p, ec) ? p : p.parent_path(), "环境变量 VELOMIND_TINYLLAMA_PATH"});
        }
    }
    if (const char* env = std::getenv("VELOMIND_MODEL_DIR")) {
        candidates.push_back({fs::path(env) / dir_name, "环境变量 VELOMIND_MODEL_DIR"});
    }

    // 2. 仓库内 models/ 目录
    auto repo_root = find_repo_root();
    candidates.push_back({repo_root / "models" / dir_name, "仓库 models/ 目录"});

    // 3. 用户缓存目录 ~/.cache/velomind/models/
    candidates.push_back({get_user_cache_dir() / dir_name, "用户缓存 ~/.cache/velomind/models/"});

    // 4. 系统遗留兼容路径
    candidates.push_back({fs::path("/home/aiza/workspace/assets/ai-models") / dir_name, "系统兼容路径"});

    // 逐级探查有效资产
    for (const auto& entry : candidates) {
        if (!fs::exists(entry.path, ec) || !fs::is_directory(entry.path, ec)) {
            continue;
        }

        fs::path model_file = entry.path / "model.safetensors";
        fs::path tok_file = entry.path / tok_name;

        // TinyLlama 在缺少 tokenizer.model 时尝试 tokenizer.json 回退
        if (family == ModelFamily::TinyLlama && !fs::exists(tok_file, ec)) {
            fs::path tok_json = entry.path / "tokenizer.json";
            if (fs::exists(tok_json, ec)) {
                tok_file = tok_json;
            }
        }

        if (fs::exists(model_file, ec) && fs::exists(tok_file, ec)) {
            res.model_dir = fs::canonical(entry.path, ec);
            res.model_path = fs::canonical(model_file, ec);
            res.tokenizer_path = fs::canonical(tok_file, ec);

            fs::path cfg_file = entry.path / "config.json";
            if (fs::exists(cfg_file, ec)) {
                res.config_path = fs::canonical(cfg_file, ec);
            }

            res.found = true;
            res.source_desc = entry.desc;
            return res;
        }
    }

    return res;
}

auto ensure_model_assets_or_prompt(
    ModelFamily family,
    ModelAssetPaths& assets,
    bool auto_download
) -> bool {
    if (assets.found) {
        return true;
    }

    if (const char* env_auto = std::getenv("VELOMIND_AUTO_DOWNLOAD")) {
        std::string s(env_auto);
        if (s == "1" || s == "true" || s == "TRUE" || s == "ON" || s == "yes") {
            auto_download = true;
        }
    }

    const std::string model_name = (family == ModelFamily::SmolLM2) ? "smollm2" : "tinyllama";
    const std::string display_name = (family == ModelFamily::SmolLM2) ? "SmolLM2-135M" : "TinyLlama_v1.1";
    const std::string size_str = (family == ModelFamily::SmolLM2) ? "约 260 MB" : "约 4.1 GB";

    bool is_interactive_tty = (isatty(STDIN_FILENO) != 0 && isatty(STDOUT_FILENO) != 0);

    if (!auto_download && is_interactive_tty) {
        std::cout << "\n======================================================================\n"
                  << " [VeloMind] 本地未检测到 " << display_name << " 模型权重（" << size_str << "）。\n"
                  << " 是否立即自动下载至仓库 models/ 目录？[Y/n]: " << std::flush;

        std::string input;
        if (!std::getline(std::cin, input)) {
            return false;
        }
        if (!input.empty() && input[0] != 'y' && input[0] != 'Y') {
            return false;
        }
        auto_download = true;
    }

    if (!auto_download) {
        return false;
    }

    auto repo_root = find_repo_root();
    fs::path script_path = repo_root / "scripts" / "download_model.sh";
    std::error_code ec;
    if (!fs::exists(script_path, ec)) {
        script_path = "scripts/download_model.sh";
    }

    std::string cmd = "bash \"" + script_path.string() + "\" " + model_name;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[VeloMind] 模型下载工具退出异常，退出码: " << ret << "\n";
        return false;
    }

    assets = resolve_model_assets(family);
    return assets.found;
}

void print_missing_model_guide(ModelFamily family, const std::string& program_name) {
    const std::string model_arg = (family == ModelFamily::SmolLM2) ? "smollm2" : "tinyllama";
    const std::string display_name = (family == ModelFamily::SmolLM2) ? "SmolLM2-135M" : "TinyLlama_v1.1";
    const std::string size_str = (family == ModelFamily::SmolLM2) ? "约 260 MB" : "约 4.1 GB";

    std::cerr << "\n======================================================================\n"
              << "[VeloMind] 运行失败：未找到 " << display_name << " 模型资产。\n\n"
              << "快速启动解决方案：\n"
              << "  1. 一键下载模型权重（" << size_str << "）：\n"
              << "     ./scripts/download_model.sh " << model_arg << "\n"
              << "     （若连接受阻可添加 '--hf-mirror' 使用国内高速镜像）\n\n"
              << "  2. 使用合成随机权重直接体验全管线推理（零下载，开箱即跑）：\n"
              << "     " << program_name << " --synthetic\n\n"
              << "  3. 指定已有模型与分词器路径：\n"
              << "     " << program_name << " --model <path/to/model.safetensors> --tokenizer <path/to/tokenizer>\n"
              << "======================================================================\n\n";
}

} // namespace velomind::examples::llama
