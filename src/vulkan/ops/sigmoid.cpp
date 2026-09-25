#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/vulkan_backend.h"


namespace velomind::backend::vulkan {

namespace {

    struct SigmoidPipeline {
        VkShaderModule         shader      = VK_NULL_HANDLE;
        VkDescriptorSetLayout  set_layout  = VK_NULL_HANDLE;
        VkPipelineLayout       pipe_layout = VK_NULL_HANDLE;
        VkPipeline             pipeline    = VK_NULL_HANDLE;
        VkDescriptorPool       desc_pool   = VK_NULL_HANDLE;
        bool                   ok          = false;
    };

    auto sigmoid_pipeline() -> SigmoidPipeline& {
        static SigmoidPipeline p;
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!backend_vulkan::ensure_vulkan_ready()) return;
            auto& state = backend_vulkan::vulkan_state();

            auto bytes = backend_vulkan::load_shader_spv("sigmoid");
            if (bytes.empty()) return;

            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes.size();
            smci.pCode    = reinterpret_cast<const uint32_t*>(bytes.data());
            if (vkCreateShaderModule(state.device, &smci, nullptr, &p.shader)
                != VK_SUCCESS) return;

            std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
                VkDescriptorSetLayoutBinding{
                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                VkDescriptorSetLayoutBinding{
                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo dslci{};
            dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dslci.bindingCount = bindings.size();
            dslci.pBindings    = bindings.data();
            if (vkCreateDescriptorSetLayout(state.device, &dslci, nullptr,
                                            &p.set_layout) != VK_SUCCESS) return;

            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset     = 0;
            pcr.size       = sizeof(uint32_t);

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts    = &p.set_layout;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
            if (vkCreatePipelineLayout(state.device, &plci, nullptr,
                                       &p.pipe_layout) != VK_SUCCESS) return;

            VkComputePipelineCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = p.shader;
            cpci.stage.pName  = "main";
            cpci.layout       = p.pipe_layout;
            if (vkCreateComputePipelines(state.device, VK_NULL_HANDLE, 1,
                                         &cpci, nullptr,
                                         &p.pipeline) != VK_SUCCESS) return;

            VkDescriptorPoolSize pool_size{};
            pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = 512 * 2;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.maxSets       = 512;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes    = &pool_size;
            dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            if (vkCreateDescriptorPool(state.device, &dpci, nullptr,
                                       &p.desc_pool) != VK_SUCCESS) return;

            p.ok = true;
        });
        return p;
    }

    inline auto buffer_of(const TensorStorage* s) -> VkBuffer {
        struct VulkanExt { VkBuffer buffer; VkDeviceMemory memory; void* mapped; };
        return reinterpret_cast<const VulkanExt*>(s + 1)->buffer;
    }

    void sigmoid_impl(const TensorStorage* const* in, TensorStorage* const* out,
                      const void* ) {
        auto& state = backend_vulkan::vulkan_state();
        auto& pipe  = sigmoid_pipeline();
        if (!pipe.ok)
            throw std::runtime_error(vulkan_error_context(Op::Sigmoid, *out[0], "pipeline") +
                                     ": initialization failed");
        check_vulkan(VK_SUCCESS, Op::Sigmoid, *out[0], "pipeline");

        const std::size_t n = storage_numel(*out[0]);
        if (n == 0) return;

        VulkanDispatchGuard guard{state, pipe.desc_pool};
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = pipe.desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &pipe.set_layout;
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkResult desc_alloc_res = VK_SUCCESS;
        {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            desc_alloc_res = vkAllocateDescriptorSets(state.device, &dsai, &desc_set);
        }
        check_vulkan(desc_alloc_res, Op::Sigmoid, *out[0], "descriptor allocation");
        guard.descriptor_set = desc_set;

        std::array<VkDescriptorBufferInfo, 2> buffer_infos = {
            VkDescriptorBufferInfo{ buffer_of(in[0]),  in[0]->offset_bytes, n * sizeof(float) },
            VkDescriptorBufferInfo{ buffer_of(out[0]), out[0]->offset_bytes, n * sizeof(float) },
        };
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (std::size_t i = 0; i < writes.size(); ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = desc_set;
            writes[i].dstBinding      = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &buffer_infos[i];
        }
        vkUpdateDescriptorSets(state.device, writes.size(), writes.data(),
                               0, nullptr);

        const uint32_t n_u32 = static_cast<uint32_t>(n);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = guard.command_pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkResult cmd_alloc_res = VK_SUCCESS;
        {
            std::lock_guard<std::mutex> lock(backend_vulkan::vulkan_pool_mutex());
            cmd_alloc_res = vkAllocateCommandBuffers(state.device, &cbai, &guard.command);
        }
        check_vulkan(cmd_alloc_res, Op::Sigmoid, *out[0], "command allocation");
        VkCommandBuffer cmd = guard.command;

        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vulkan(vkBeginCommandBuffer(cmd, &cbbi),
                     Op::Sigmoid, *out[0], "command begin");

        maybe_insert_compute_barrier(cmd);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.pipe_layout, 0, 1, &desc_set,
                                0, nullptr);
        vkCmdPushConstants(cmd, pipe.pipe_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(n_u32), &n_u32);
        const std::size_t groups = (n + 63) / 64;
        vkCmdDispatch(cmd, static_cast<uint32_t>(groups), 1, 1);
        guard.finish(Op::Sigmoid, *out[0]);
    }

VELOMIND_REGISTER_UNARY_FN(DeviceType::VULKAN, Op::Sigmoid, DataType::Float32, DataType::Float32, &sigmoid_impl);

}

}
