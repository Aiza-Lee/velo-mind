#include <array>
#include <cstddef>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "internal/vulkan_backend.h"

namespace velomind::backend_vulkan {

namespace {

    thread_local VulkanBatchContext* active_vulkan_batch = nullptr;

    class FencePool {
    public:
        ~FencePool() {
            cleanup();
        }

        void cleanup() {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& s = vulkan_state();
            if (s.device == VK_NULL_HANDLE) return;
            for (auto f : pool_) {
                if (f != VK_NULL_HANDLE) vkDestroyFence(s.device, f, nullptr);
            }
            pool_.clear();
        }

        auto acquire(VkResult* res_out) -> VkFence {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& s = vulkan_state();
            if (!pool_.empty()) {
                VkFence f = pool_.back();
                pool_.pop_back();
                VkResult res = vkResetFences(s.device, 1, &f);
                if (res_out) *res_out = res;
                return f;
            }
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence f = VK_NULL_HANDLE;
            VkResult res = vkCreateFence(s.device, &fci, nullptr, &f);
            if (res_out) *res_out = res;
            return f;
        }

        void release(VkFence f) {
            if (f == VK_NULL_HANDLE) return;
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push_back(f);
        }

    private:
        std::mutex           mutex_;
        std::vector<VkFence> pool_;
    };

    auto global_fence_pool() -> FencePool& {
        static FencePool fp;
        return fp;
    }

} // namespace

auto vulkan_state() -> VulkanState& {
    static VulkanState s;
    return s;
}

auto vulkan_mutex() -> std::mutex& {
    static std::mutex m;
    return m;
}

auto vulkan_queue_mutex() -> std::mutex& {
    static std::mutex m;
    return m;
}

auto vulkan_pool_mutex() -> std::mutex& {
    static std::mutex m;
    return m;
}

auto pick_physical_device(VkInstance instance) -> VkPhysicalDevice {
    std::uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    // 优先选择离散独立显卡
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            return dev;
        }
    }
    return devices[0];
}

auto find_compute_queue_family(VkPhysicalDevice physical) -> uint32_t {
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    for (std::uint32_t i = 0; i < family_count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    return UINT32_MAX;
}

auto ensure_vulkan_ready() -> bool {
    auto& s = vulkan_state();
    if (s.ready) return true;

    std::lock_guard lock(vulkan_mutex());
    if (s.ready) return true;

    VkApplicationInfo app_info{};
    app_info.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "velomind-core";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName      = "velomind";
    app_info.engineVersion    = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion       = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ic_info{};
    ic_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&ic_info, nullptr, &s.instance) != VK_SUCCESS) {
        s.instance = VK_NULL_HANDLE;
        return false;
    }

    s.physical = pick_physical_device(s.instance);
    if (s.physical == VK_NULL_HANDLE) return false;

    s.queue_family = find_compute_queue_family(s.physical);
    if (s.queue_family == UINT32_MAX) return false;

    constexpr float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qc_info{};
    qc_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qc_info.queueFamilyIndex = s.queue_family;
    qc_info.queueCount = 1;
    qc_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dc_info{};
    dc_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc_info.queueCreateInfoCount = 1;
    dc_info.pQueueCreateInfos = &qc_info;

    if (vkCreateDevice(s.physical, &dc_info, nullptr, &s.device) != VK_SUCCESS) {
        s.device = VK_NULL_HANDLE;
        return false;
    }

    vkGetDeviceQueue(s.device, s.queue_family, 0, &s.queue);

    VkCommandPoolCreateInfo cp_info{};
    cp_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_info.queueFamilyIndex = s.queue_family;
    // 启用 RESET_COMMAND_BUFFER 以支持命令缓冲区从池中取出后独立重置复用
    cp_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(s.device, &cp_info, nullptr, &s.command_pool) != VK_SUCCESS) {
        s.command_pool = VK_NULL_HANDLE;
        return false;
    }

    vkGetPhysicalDeviceMemoryProperties(s.physical, &s.mem_props);
    s.ready = true;
    return true;
}

auto find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags flags) -> uint32_t {
    auto& s = vulkan_state();
    for (std::uint32_t i = 0; i < s.mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) continue;
        if ((s.mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

auto find_host_visible_coherent_type(uint32_t type_bits) -> uint32_t {
    return find_memory_type(type_bits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

auto find_host_cached_type(uint32_t type_bits) -> uint32_t {
    return find_memory_type(type_bits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
}

auto find_device_local_type(uint32_t type_bits) -> uint32_t {
    return find_memory_type(type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

struct ThreadCommandPoolHolder {
    VkCommandPool pool = VK_NULL_HANDLE;

    ~ThreadCommandPoolHolder() {
        if (pool != VK_NULL_HANDLE) {
            auto& s = vulkan_state();
            if (s.device != VK_NULL_HANDLE) {
                std::lock_guard<std::mutex> lock(vulkan_mutex());
                vkDestroyCommandPool(s.device, pool, nullptr);
            }
            pool = VK_NULL_HANDLE;
        }
    }
};

auto get_thread_command_pool() -> VkCommandPool {
    static thread_local ThreadCommandPoolHolder holder;
    if (holder.pool == VK_NULL_HANDLE) {
        if (!ensure_vulkan_ready()) return VK_NULL_HANDLE;
        auto& s = vulkan_state();
        if (s.device == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = s.queue_family;
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        std::lock_guard<std::mutex> lock(vulkan_mutex());
        if (vkCreateCommandPool(s.device, &cpci, nullptr, &holder.pool) != VK_SUCCESS) {
            return s.command_pool;
        }
    }
    return holder.pool;
}

auto acquire_pooled_fence(VkResult* res_out) -> VkFence {
    return global_fence_pool().acquire(res_out);
}

void release_pooled_fence(VkFence fence) {
    global_fence_pool().release(fence);
}

auto get_active_vulkan_batch() -> VulkanBatchContext* {
    return active_vulkan_batch;
}

void set_active_vulkan_batch(VulkanBatchContext* ctx) {
    active_vulkan_batch = ctx;
}

void VulkanBatchContext::flush_and_sync() {
    if (!active || recorded_commands.empty()) return;

    auto& s = vulkan_state();
    VkResult fence_res = VK_SUCCESS;
    VkFence fence = acquire_pooled_fence(&fence_res);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = static_cast<uint32_t>(recorded_commands.size());
    si.pCommandBuffers    = recorded_commands.data();

    {
        std::lock_guard<std::mutex> lock(vulkan_queue_mutex());
        vkQueueSubmit(s.queue, 1, &si, fence);
    }
    vkWaitForFences(s.device, 1, &fence, VK_TRUE, UINT64_MAX);
    release_pooled_fence(fence);

    auto thread_pool = get_thread_command_pool();
    {
        std::lock_guard<std::mutex> lock(vulkan_pool_mutex());
        if (s.device != VK_NULL_HANDLE && thread_pool != VK_NULL_HANDLE && !recorded_commands.empty()) {
            vkFreeCommandBuffers(s.device, thread_pool,
                                 static_cast<uint32_t>(recorded_commands.size()),
                                 recorded_commands.data());
        }
        for (const auto& [dp, ds] : recorded_sets) {
            if (dp != VK_NULL_HANDLE && ds != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(s.device, dp, 1, &ds);
            }
        }
    }
    recorded_commands.clear();
    recorded_sets.clear();
}

ScopedVulkanBatch::ScopedVulkanBatch() {
    if (!is_vulkan_batch_enabled() || !ensure_vulkan_ready()) return;
    prev_ = get_active_vulkan_batch();
    ctx_.active = true;
    set_active_vulkan_batch(&ctx_);
}

ScopedVulkanBatch::~ScopedVulkanBatch() {
    if (ctx_.active) {
        ctx_.flush_and_sync();
        ctx_.active = false;
    }
    set_active_vulkan_batch(prev_);
}

} // namespace velomind::backend_vulkan
