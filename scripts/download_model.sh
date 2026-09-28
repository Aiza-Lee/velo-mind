#!/usr/bin/env bash
# VeloMind - 模型权重下载脚本
# 支持 SmolLM2-135M 与 TinyLlama 模型的轻量自动化下载与缓存

set -euo pipefail

MODEL="smollm2"
TARGET_DIR=""
USE_CACHE=false
MIRROR=""
HELP=false

# 脚本所在目录及仓库根目录推导
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat << EOF
VeloMind 模型下载工具

用法:
  $(basename "$0") [模型名称] [选项]

支持模型:
  smollm2     SmolLM2-135M (推荐，约 260 MB，下载快，CPU/GPU 流畅)
  tinyllama   TinyLlama_v1.1 (约 4.1 GB，标准 Llama 架构)

选项:
  --dir <路径>       指定模型保存的目标目录
  --cache            下载至用户标准缓存目录 (~/.cache/velomind/models/...)
  --mirror <URL>     指定 Hugging Face 镜像源 (如 https://hf-mirror.com)
  --hf-mirror        直接使用国内高速镜像源 https://hf-mirror.com
  --official         强制使用 Hugging Face 官方源 https://huggingface.co
  -h, --help         显示帮助信息

示例:
  # 一键下载默认推荐模型 SmolLM2 到仓库 models/ 目录
  $(basename "$0") smollm2

  # 使用国内镜像源下载到用户全局缓存
  $(basename "$0") smollm2 --hf-mirror --cache
EOF
}

# 参数解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        smollm2|smollm2-135m|SmolLM2|SmolLM2-135M)
            MODEL="smollm2"
            shift
            ;;
        tinyllama|tinyllama-1.1b|TinyLlama|TinyLlama_v1.1)
            MODEL="tinyllama"
            shift
            ;;
        --dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --cache)
            USE_CACHE=true
            shift
            ;;
        --mirror)
            MIRROR="$2"
            shift 2
            ;;
        --hf-mirror)
            MIRROR="https://hf-mirror.com"
            shift
            ;;
        --official)
            MIRROR="https://huggingface.co"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[velomind Error] 未知参数: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# 确定镜像基础 URL
if [[ -z "${MIRROR}" ]]; then
    if [[ -n "${HF_ENDPOINT:-}" ]]; then
        BASE_URL="${HF_ENDPOINT}"
    else
        BASE_URL="https://huggingface.co"
    fi
else
    BASE_URL="${MIRROR}"
fi
# 去除 URL 尾部斜杠
BASE_URL="${BASE_URL%/}"

# 模型定义与文件清单
case "${MODEL}" in
    smollm2)
        HF_REPO="HuggingFaceTB/SmolLM2-135M"
        DIR_NAME="SmolLM2-135M"
        FILES=("config.json" "tokenizer.json" "model.safetensors")
        DISPLAY_NAME="SmolLM2-135M (~260 MB)"
        RUN_HINT="./build/release/examples/smollm2/smollm2_chat"
        ;;
    tinyllama)
        HF_REPO="TinyLlama/TinyLlama_v1.1"
        DIR_NAME="TinyLlama_v1.1"
        FILES=("config.json" "tokenizer.model" "model.safetensors")
        DISPLAY_NAME="TinyLlama_v1.1 (~4.1 GB)"
        RUN_HINT="./build/release/examples/tinyllama/tinyllama_chat"
        ;;
    *)
        echo "[velomind Error] 不受支持的模型: ${MODEL}" >&2
        exit 1
        ;;
esac

# 确定目标目录
if [[ -n "${TARGET_DIR}" ]]; then
    DEST_DIR="${TARGET_DIR}"
elif [[ "${USE_CACHE}" == true ]]; then
    CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/velomind/models"
    DEST_DIR="${CACHE_ROOT}/${DIR_NAME}"
else
    # 默认优先放入仓库下的 models/ 目录
    DEST_DIR="${REPO_ROOT}/models/${DIR_NAME}"
fi

mkdir -p "${DEST_DIR}"

echo "======================================================================"
echo " VeloMind 模型资产下载工具"
echo " 模型: ${DISPLAY_NAME}"
echo " 来源: ${BASE_URL}/${HF_REPO}"
echo " 目标: ${DEST_DIR}"
echo "======================================================================"

# 检查下载工具
DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    echo "[velomind Error] 系统未找到 curl 或 wget，请先安装下载工具。" >&2
    exit 1
fi

# 下载单个文件函数
download_file() {
    local filename="$1"
    local url="${BASE_URL}/${HF_REPO}/resolve/main/${filename}"
    local dest_file="${DEST_DIR}/${filename}"
    local part_file="${dest_file}.part"

    # 若文件已存在且大小大于 0，进行简易检查
    if [[ -f "${dest_file}" && -s "${dest_file}" ]]; then
        echo "  [已就绪] ${filename} (跳过下载)"
        return 0
    fi

    echo "  [正在下载] ${filename} ..."
    rm -f "${part_file}"

    if [[ "${DOWNLOADER}" == "curl" ]]; then
        local curl_opts=("-L" "--fail" "--retry" "3" "--retry-delay" "2")
        if [[ -t 1 ]]; then
            curl_opts+=("--progress-bar")
        else
            curl_opts+=("-s")
        fi
        if ! curl "${curl_opts[@]}" -o "${part_file}" "${url}"; then
            echo "[velomind Error] 下载失败: ${url}" >&2
            echo "  提示: 若遇到网络超时或连接受阻，可尝试添加 '--hf-mirror' 使用国内镜像源。" >&2
            rm -f "${part_file}"
            return 1
        fi
    else
        local wget_opts=("-q" "--show-progress" "--tries=3")
        if ! wget "${wget_opts[@]}" -O "${part_file}" "${url}"; then
            echo "[velomind Error] 下载失败: ${url}" >&2
            echo "  提示: 若遇到网络超时或连接受阻，可尝试添加 '--hf-mirror' 使用国内镜像源。" >&2
            rm -f "${part_file}"
            return 1
        fi
    fi

    # 校验并原子重命名
    if [[ -f "${part_file}" && -s "${part_file}" ]]; then
        mv "${part_file}" "${dest_file}"
        echo "  [下载完成] ${filename}"
    else
        echo "[velomind Error] 下载文件为空: ${filename}" >&2
        rm -f "${part_file}"
        return 1
    fi
}

for file in "${FILES[@]}"; do
    download_file "${file}"
done

echo ""
echo "======================================================================"
echo " 模型资产准备就绪！"
echo " 路径: ${DEST_DIR}"
echo ""
echo " 启动推理示例:"
echo "   ${RUN_HINT}"
echo "======================================================================"
