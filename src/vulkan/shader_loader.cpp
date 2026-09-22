#include "internal/vulkan_backend.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__unix__)
#include <unistd.h>
#endif

namespace velomind::backend_vulkan {

namespace fs = std::filesystem;

namespace {

auto read_file_bytes(const fs::path& p) -> std::vector<char> {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) {
        return {};
    }
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto size = f.tellg();
    if (size <= 0) return {};
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<std::size_t>(size));
    if (f.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

auto get_executable_dir() -> fs::path {
#if defined(__linux__)
    char buf[1024];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    return {};
}

} // namespace

auto load_shader_spv(std::string_view shader_name) -> std::vector<char> {
    const std::string filename = std::string(shader_name) + ".spv";

    // 优先检查环境变量 VELOMIND_VULKAN_SHADER_DIR
    if (const char* env_dir = std::getenv("VELOMIND_VULKAN_SHADER_DIR")) {
        if (*env_dir != '\0') {
            const fs::path p = fs::path(env_dir) / filename;
            auto bytes = read_file_bytes(p);
            if (!bytes.empty()) return bytes;
        }
    }

    // 检查可执行文件相对路径（适应安装至 prefix/bin 与 prefix/share 结构）
    const auto exe_dir = get_executable_dir();
    if (!exe_dir.empty()) {
        const std::array<fs::path, 4> relative_paths = {
            exe_dir / "shaders" / filename,
            exe_dir / ".." / "share" / "velomind" / "shaders" / filename,
            exe_dir / ".." / "shaders" / filename,
            exe_dir / filename
        };
        for (const auto& p : relative_paths) {
            auto bytes = read_file_bytes(p);
            if (!bytes.empty()) return bytes;
        }
    }

    // 检查当前工作目录相对路径
    {
        const std::array<fs::path, 3> cwd_paths = {
            fs::path("shaders") / filename,
            fs::path("share") / "velomind" / "shaders" / filename,
            fs::path(filename)
        };
        for (const auto& p : cwd_paths) {
            auto bytes = read_file_bytes(p);
            if (!bytes.empty()) return bytes;
        }
    }

    // 检查构建期固化的安装路径
#if defined(VELOMIND_VULKAN_SHADER_INSTALL_DIR)
    {
        const fs::path p = fs::path(VELOMIND_VULKAN_SHADER_INSTALL_DIR) / filename;
        auto bytes = read_file_bytes(p);
        if (!bytes.empty()) return bytes;
    }
#endif

    // 检查构建期源码树输出目录
#if defined(VELOMIND_VULKAN_SHADER_BUILD_DIR)
    {
        const fs::path p = fs::path(VELOMIND_VULKAN_SHADER_BUILD_DIR) / filename;
        auto bytes = read_file_bytes(p);
        if (!bytes.empty()) return bytes;
    }
#endif

    return {};
}

} // namespace velomind::backend_vulkan
