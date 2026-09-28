#pragma once

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace velomind_test {

// 检查是否处于模型专用流水线环境（严格模式：资产缺失视作测试失败而非跳过）
inline auto is_model_pipeline_required() -> bool {
    const char* val = std::getenv("VELOMIND_REQUIRE_MODEL_ASSETS");
    if (!val) val = std::getenv("VELOMIND_STRICT_MODEL_TESTS");
    if (!val) return false;
    std::string s(val);
    return s == "1" || s == "true" || s == "TRUE" || s == "ON" || s == "on";
}

#ifndef VELOMIND_PROJECT_ROOT
#define VELOMIND_PROJECT_ROOT ""
#endif

// 解析相对仓库根目录的资产路径，支持从当前目录、父目录或宏定义寻址
inline auto resolve_repo_path(const std::filesystem::path& rel_path) -> std::filesystem::path {
    if (rel_path.is_absolute()) {
        return rel_path;
    }
    if (std::filesystem::exists(rel_path)) {
        return rel_path;
    }
    const std::string root_def = VELOMIND_PROJECT_ROOT;
    if (!root_def.empty()) {
        auto p = std::filesystem::path(root_def) / rel_path;
        if (std::filesystem::exists(p)) return p;
    }
    std::filesystem::path prefix = "..";
    for (int i = 0; i < 4; ++i) {
        auto candidate = prefix / rel_path;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
        prefix /= "..";
    }
    return rel_path;
}

// 获取模型资产根目录（多级级联：环境变量 > 仓库 models/ > 用户缓存 ~/.cache/velomind/models > 系统兼容目录）
inline auto get_model_assets_root() -> std::filesystem::path {
    if (const char* env = std::getenv("VELOMIND_MODEL_DIR")) {
        return std::filesystem::path(env);
    }
    auto repo_models = resolve_repo_path("models");
    if (std::filesystem::exists(repo_models)) {
        return repo_models;
    }
    if (const char* home = std::getenv("HOME")) {
        auto user_cache = std::filesystem::path(home) / ".cache" / "velomind" / "models";
        if (std::filesystem::exists(user_cache)) {
            return user_cache;
        }
    }
    auto sys_path = std::filesystem::path("/home/aiza/workspace/assets/ai-models");
    if (std::filesystem::exists(sys_path)) {
        return sys_path;
    }
    return sys_path;
}

// 获取 TinyLlama 模型 safetensors 文件路径（支持环境变量、仓库 models/、用户缓存与系统兼容路径）
inline auto get_tinyllama_model_path() -> std::filesystem::path {
    if (const char* env = std::getenv("VELOMIND_TINYLLAMA_PATH")) {
        return std::filesystem::path(env);
    }
    if (const char* env = std::getenv("VELOMIND_TINYLLAMA_DIR")) {
        return std::filesystem::path(env) / "model.safetensors";
    }
    if (const char* env = std::getenv("VELOMIND_MODEL_DIR")) {
        auto p = std::filesystem::path(env) / "TinyLlama_v1.1" / "model.safetensors";
        if (std::filesystem::exists(p)) return p;
    }
    auto repo_file = resolve_repo_path("models/TinyLlama_v1.1/model.safetensors");
    if (std::filesystem::exists(repo_file)) return repo_file;

    if (const char* home = std::getenv("HOME")) {
        auto cache_file = std::filesystem::path(home) / ".cache" / "velomind" / "models" / "TinyLlama_v1.1" / "model.safetensors";
        if (std::filesystem::exists(cache_file)) return cache_file;
    }

    auto sys_file = std::filesystem::path("/home/aiza/workspace/assets/ai-models/TinyLlama_v1.1/model.safetensors");
    if (std::filesystem::exists(sys_file)) return sys_file;

    return get_model_assets_root() / "TinyLlama_v1.1" / "model.safetensors";
}

// 获取 SmolLM2 模型目录（支持环境变量、仓库 models/、用户缓存与系统兼容路径）
inline auto get_smollm2_model_dir() -> std::filesystem::path {
    if (const char* env = std::getenv("VELOMIND_SMOLLM2_DIR")) {
        return std::filesystem::path(env);
    }
    if (const char* env = std::getenv("VELOMIND_SMOLLM2_PATH")) {
        auto p = std::filesystem::path(env);
        return std::filesystem::is_directory(p) ? p : p.parent_path();
    }
    if (const char* env = std::getenv("VELOMIND_MODEL_DIR")) {
        auto p = std::filesystem::path(env) / "SmolLM2-135M";
        if (std::filesystem::exists(p)) return p;
    }
    auto repo_dir = resolve_repo_path("models/SmolLM2-135M");
    if (std::filesystem::exists(repo_dir)) return repo_dir;

    if (const char* home = std::getenv("HOME")) {
        auto cache_dir = std::filesystem::path(home) / ".cache" / "velomind" / "models" / "SmolLM2-135M";
        if (std::filesystem::exists(cache_dir)) return cache_dir;
    }

    auto sys_dir = std::filesystem::path("/home/aiza/workspace/assets/ai-models/SmolLM2-135M");
    if (std::filesystem::exists(sys_dir)) return sys_dir;

    return get_model_assets_root() / "SmolLM2-135M";
}

// 检查模型资产文件是否存在：流水线严格要求时报错 FAIL，普通环境显式调用 Catch2 SKIP 并标注原因分类
inline auto require_or_skip_asset(const std::filesystem::path& path, const std::string& asset_desc) -> std::filesystem::path {
    auto resolved = resolve_repo_path(path);
    if (std::filesystem::exists(resolved)) return resolved;

    const std::string reason = "[ASSET_MISSING] " + asset_desc + " (path: " + path.string() + ")";
    if (is_model_pipeline_required()) {
        FAIL("模型专用流水线要求资产必须存在: " + reason);
    } else {
        SKIP("跳过模型测试: " + reason);
    }
    return resolved;
}

// 检查数组内每个数值必须有限；若出现 NaN 或 Inf 则立即以详细索引和数值报告失败
template <typename T>
inline void require_all_finite(const std::vector<T>& data, const std::string& tensor_name) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            FAIL("数值检查失败: " + tensor_name + "[" + std::to_string(i) + "] 为非有限值 (NaN/Inf)");
        }
    }
}

template <typename T>
inline void require_all_finite(const T* data, std::size_t size, const std::string& tensor_name) {
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            FAIL("数值检查失败: " + tensor_name + "[" + std::to_string(i) + "] 为非有限值 (NaN/Inf)");
        }
    }
}

} // namespace velomind_test
