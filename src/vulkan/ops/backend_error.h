#pragma once

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

#include "internal/vulkan_backend.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"

namespace velomind::backend::vulkan {

inline auto vulkan_error_context(Op op, const TensorStorage& output, const char* stage)
    -> std::string {
    std::string message = std::string(op_name(op)) + " on Vulkan output shape=[";
    for (std::size_t i = 0; i < output.shape.size(); ++i) {
        if (i) message += ',';
        message += std::to_string(output.shape[i]);
    }
    return message + "] at " + stage;
}

inline void check_vulkan(VkResult result, Op op, const TensorStorage& output,
                         const char* stage) {
    const auto context = vulkan_error_context(op, output, stage);
    if (result != VK_SUCCESS)
        throw std::runtime_error(context + ": VkResult " +
                                 std::to_string(static_cast<int>(result)));
    if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
        const auto key = std::string("Vulkan:") + op_name(op) + ':' + stage;
        if (key == injected) throw std::runtime_error(context + ": injected failure");
    }
}

// 在批处理命令间插入计算流水线内存屏障，保证前序写对后序读一致
inline void maybe_insert_compute_barrier(VkCommandBuffer cmd) {
    auto* batch = backend_vulkan::get_active_vulkan_batch();
    if (batch && batch->has_pending()) {
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
}

struct VulkanDispatchGuard {
    backend_vulkan::VulkanState& state;
    VkCommandPool    command_pool    = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet  descriptor_set  = VK_NULL_HANDLE;
    VkCommandBuffer  command         = VK_NULL_HANDLE;
    VkFence          fence           = VK_NULL_HANDLE;
    bool             submitted       = false;
    bool             in_batch        = false;

    VulkanDispatchGuard(backend_vulkan::VulkanState& s, VkDescriptorPool dp)
        : state(s), descriptor_pool(dp) {
        command_pool = backend_vulkan::get_thread_command_pool();
        auto* batch = backend_vulkan::get_active_vulkan_batch();
        in_batch = (batch && batch->active);
    }

    ~VulkanDispatchGuard() {
        if (in_batch && command == VK_NULL_HANDLE) {
            // 已转交 batch 队列统一在边界提交释放
            return;
        }
        if (submitted) {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_queue_mutex());
            vkQueueWaitIdle(state.queue);
        }
        if (fence != VK_NULL_HANDLE) {
            backend_vulkan::release_pooled_fence(fence);
        }
        if (command != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            vkFreeCommandBuffers(state.device, command_pool, 1, &command);
        }
        if (descriptor_pool != VK_NULL_HANDLE && descriptor_set != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            vkFreeDescriptorSets(state.device, descriptor_pool, 1, &descriptor_set);
        }
    }

    void finish(Op op, const TensorStorage& output) {
        check_vulkan(vkEndCommandBuffer(command), op, output, "command end");

        auto* batch = backend_vulkan::get_active_vulkan_batch();
        if (in_batch && batch && batch->active) {
            batch->recorded_commands.push_back(command);
            batch->recorded_sets.push_back({descriptor_pool, descriptor_set});
            command = VK_NULL_HANDLE;
            descriptor_set = VK_NULL_HANDLE;
            descriptor_pool = VK_NULL_HANDLE;
            return;
        }

        VkResult fence_res = VK_SUCCESS;
        fence = backend_vulkan::acquire_pooled_fence(&fence_res);
        check_vulkan(fence_res, op, output, "fence create");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &command;

        VkResult submit_result = VK_SUCCESS;
        {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_queue_mutex());
            submit_result = vkQueueSubmit(state.queue, 1, &si, fence);
        }
        submitted = (submit_result == VK_SUCCESS);
        check_vulkan(submit_result, op, output, "submit");

        const auto wait_result = vkWaitForFences(state.device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (wait_result == VK_SUCCESS) submitted = false;
        check_vulkan(wait_result, op, output, "fence wait");

        backend_vulkan::release_pooled_fence(fence);
        fence = VK_NULL_HANDLE;

        if (command != VK_NULL_HANDLE && command_pool != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            vkFreeCommandBuffers(state.device, command_pool, 1, &command);
            command = VK_NULL_HANDLE;
        }

        VkResult reset_result = VK_SUCCESS;
        if (descriptor_pool != VK_NULL_HANDLE && descriptor_set != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            reset_result = vkFreeDescriptorSets(state.device, descriptor_pool, 1, &descriptor_set);
            descriptor_set = VK_NULL_HANDLE;
            descriptor_pool = VK_NULL_HANDLE;
        }
        check_vulkan(reset_result, op, output, "descriptor reset");
    }
};

} // namespace velomind::backend::vulkan
