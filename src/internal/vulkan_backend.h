#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <vector>

#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind::backend_vulkan {

// Vulkan 扩展数据，附着在 TensorStorage 块尾部以检索底层 VkBuffer。
struct VulkanExt {
    VkBuffer        buffer   = VK_NULL_HANDLE;
    VkDeviceMemory  memory   = VK_NULL_HANDLE;
    void*           mapped   = nullptr;
};

inline auto buffer_of(const TensorStorage* s) -> VkBuffer {
    return reinterpret_cast<const VulkanExt*>(s + 1)->buffer;
}

struct VulkanState {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physical       = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    uint32_t         queue_family   = 0;
    VkCommandPool    command_pool   = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    bool             ready          = false;
};

auto vulkan_state() -> VulkanState&;
auto ensure_vulkan_ready() -> bool;

// 全局互斥锁：保护非线程安全的 VkQueue 提交、资源池与内存操作
auto vulkan_mutex() -> std::mutex&;
auto vulkan_queue_mutex() -> std::mutex&;
auto vulkan_pool_mutex() -> std::mutex&;

// 物理内存堆与内存类型查询
auto find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags flags) -> uint32_t;
auto find_host_visible_coherent_type(uint32_t type_bits) -> uint32_t;
auto find_host_cached_type(uint32_t type_bits) -> uint32_t;
auto find_device_local_type(uint32_t type_bits) -> uint32_t;

// 获取当前线程独占的 VkCommandPool，避免跨线程并发录制命令池冲突
auto get_thread_command_pool() -> VkCommandPool;

// Fence 资源池借还接口
auto acquire_pooled_fence(VkResult* res_out = nullptr) -> VkFence;
void release_pooled_fence(VkFence fence);

// 判断当前环境是否允许命令批量录制（故障注入或调试同步时关闭）
inline auto is_vulkan_batch_enabled() -> bool {
    if (std::getenv("VELOMIND_FAIL_BACKEND")) return false;
    if (std::getenv("VELOMIND_VULKAN_SYNC")) return false;
    return true;
}

// 批量命令录制上下文
struct VulkanBatchContext {
    std::vector<VkCommandBuffer>                              recorded_commands;
    std::vector<std::pair<VkDescriptorPool, VkDescriptorSet>> recorded_sets;
    bool                                                      active = false;

    auto has_pending() const -> bool { return active && !recorded_commands.empty(); }
    void flush_and_sync();
};

auto get_active_vulkan_batch() -> VulkanBatchContext*;
void set_active_vulkan_batch(VulkanBatchContext* ctx);

// 在主机 CPU 访问或边界同步前刷新并等待待提交的 Vulkan 批次命令
inline void sync_vulkan_batch_if_pending() {
    auto* batch = get_active_vulkan_batch();
    if (batch && batch->has_pending()) {
        batch->flush_and_sync();
    }
}

// RAII 作用域守卫：在 Executable::execute 期间启用连续命令批量录制
class ScopedVulkanBatch {
public:
    ScopedVulkanBatch();
    ~ScopedVulkanBatch();

    ScopedVulkanBatch(const ScopedVulkanBatch&) = delete;
    ScopedVulkanBatch& operator=(const ScopedVulkanBatch&) = delete;

private:
    VulkanBatchContext  ctx_;
    VulkanBatchContext* prev_ = nullptr;
};

// 查找并加载指定的 SPIR-V 着色器二进制数据
// 优先检索 VELOMIND_VULKAN_SHADER_DIR，次查可执行文件相对路径、安装路径与构建路径
auto load_shader_spv(std::string_view shader_name) -> std::vector<char>;

} // namespace velomind::backend_vulkan
