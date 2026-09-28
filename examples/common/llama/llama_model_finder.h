#pragma once

#include <filesystem>
#include <string>

namespace velomind::examples::llama {

// 模型架构家族标识
enum class ModelFamily {
    SmolLM2,
    TinyLlama,
};

// 模型资产解析结果
struct ModelAssetPaths {
    std::filesystem::path model_dir;
    std::filesystem::path model_path;
    std::filesystem::path tokenizer_path;
    std::filesystem::path config_path;
    bool                  found = false;
    std::string           source_desc;
};

// 获取标准用户缓存目录（~/.cache/velomind/models）
auto get_user_cache_dir() -> std::filesystem::path;

// 探测工程根目录（优先使用编译宏定义与可执行文件相对路径）
auto find_repo_root() -> std::filesystem::path;

// 多级级联解析模型资产路径
// 优先级：显式参数 > 环境变量 > 仓库 models/ > 用户缓存 ~/.cache/velomind/models/ > 系统兼容目录
auto resolve_model_assets(
    ModelFamily family,
    const std::string& explicit_model_path = "",
    const std::string& explicit_tokenizer_path = ""
) -> ModelAssetPaths;

// 交互式提示或调用下载脚本准备资产
// 若处于 TTY 交互终端且 auto_download 为 false，向用户询问确认；若 auto_download 为 true 则直接调用脚本拉取
auto ensure_model_assets_or_prompt(
    ModelFamily family,
    ModelAssetPaths& assets,
    bool auto_download = false
) -> bool;

// 打印格式化的资产缺失排错与快速上手指引
void print_missing_model_guide(ModelFamily family, const std::string& program_name);

} // namespace velomind::examples::llama
