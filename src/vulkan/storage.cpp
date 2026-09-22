#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>

#include <vulkan/vulkan.h>

#include "internal/registry/memory_transfer.h"
#include "internal/registry/storage.h"
#include "internal/vulkan_backend.h"

namespace velomind::backend::vulkan {

namespace {

    // 扩展状态与 TensorStorage 放在同一块内存中，供算子从 storage 反查 VkBuffer。
    struct VulkanExt {
        VkBuffer        buffer   = VK_NULL_HANDLE;
        VkDeviceMemory  memory   = VK_NULL_HANDLE;
        void*           mapped   = nullptr;
    };

    inline auto ext_of(TensorStorage* s) -> VulkanExt* {
        return reinterpret_cast<VulkanExt*>(s + 1);
    }

    // 释放底层 VkBuffer 和 VkDeviceMemory，调用前需持有或自行获取 vulkan_mutex。
    void release_ext_locked(VulkanExt* ext) {
        auto& state = backend_vulkan::vulkan_state();
        if (state.device == VK_NULL_HANDLE) return;
        if (ext->buffer != VK_NULL_HANDLE) {
            if (ext->mapped != nullptr) {
                vkUnmapMemory(state.device, ext->memory);
                ext->mapped = nullptr;
            }
            vkDestroyBuffer(state.device, ext->buffer, nullptr);
            ext->buffer = VK_NULL_HANDLE;
        }
        if (ext->memory != VK_NULL_HANDLE) {
            vkFreeMemory(state.device, ext->memory, nullptr);
            ext->memory = VK_NULL_HANDLE;
        }
    }

    void release_ext(VulkanExt* ext) {
        std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_mutex());
        release_ext_locked(ext);
    }

    void destroy_block(TensorStorage* p) {
        if (p == nullptr) return;
        release_ext(ext_of(p));
        p->~TensorStorage();
        ext_of(p)->~VulkanExt();
        ::operator delete(p);
    }

    auto vulkan_allocator(std::size_t bytes) -> std::shared_ptr<TensorStorage> {
        if (!backend_vulkan::ensure_vulkan_ready()) return nullptr;

        auto& state = backend_vulkan::vulkan_state();

        auto* raw = static_cast<TensorStorage*>(
            ::operator new(sizeof(TensorStorage) + sizeof(VulkanExt)));
        new (raw) TensorStorage{};
        auto* ext = new (ext_of(raw)) VulkanExt{};

        raw->size_bytes     = bytes;
        raw->capacity_bytes = bytes;
        raw->offset_bytes   = 0;
        raw->device         = DeviceType::VULKAN;
        raw->data           = nullptr;

        if (bytes == 0) {
            return std::shared_ptr<TensorStorage>(raw, destroy_block);
        }

        std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_mutex());

        auto fail = [&]() -> std::shared_ptr<TensorStorage> {
            release_ext_locked(ext);
            raw->~TensorStorage();
            ext->~VulkanExt();
            ::operator delete(raw);
            return nullptr;
        };

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(state.device, &bci, nullptr, &ext->buffer) != VK_SUCCESS) {
            return fail();
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(state.device, ext->buffer, &req);
        // 持久映射主机可见一致性内存，在多线程下受 vulkan_mutex 保护以保证驱动堆管理安全。
        auto mem_type = backend_vulkan::find_host_visible_coherent_type(req.memoryTypeBits);
        if (mem_type == UINT32_MAX) return fail();

        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = mem_type;
        if (vkAllocateMemory(state.device, &mai, nullptr, &ext->memory) != VK_SUCCESS) {
            return fail();
        }
        if (vkBindBufferMemory(state.device, ext->buffer, ext->memory, 0) != VK_SUCCESS) {
            return fail();
        }
        if (vkMapMemory(state.device, ext->memory, 0, bytes, 0, &ext->mapped) != VK_SUCCESS) {
            return fail();
        }
        raw->data = ext->mapped;

        return std::shared_ptr<TensorStorage>(raw, destroy_block);
    }

    void vulkan_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
        if (bytes == 0 || dst == nullptr || src == nullptr) return;

        std::memcpy(dst, src, bytes);
    }

    void vulkan_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
        if (bytes == 0 || dst == nullptr || src == nullptr) return;

        std::memcpy(dst, src, bytes);
    }

    VELOMIND_REGISTER_STORAGE_CREATOR(DeviceType::VULKAN, vulkan_allocator)
    VELOMIND_REGISTER_MEMORY_TRANSFER(DeviceType::VULKAN, vulkan_memcpy_h2d, vulkan_memcpy_d2h)

}

}
