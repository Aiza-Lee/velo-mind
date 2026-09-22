#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/vulkan_backend.h"


namespace velomind::backend::vulkan {

namespace {

    struct MatMulPipeline {
        VkShaderModule         shader      = VK_NULL_HANDLE;
        VkDescriptorSetLayout  set_layout  = VK_NULL_HANDLE;
        VkPipelineLayout       pipe_layout = VK_NULL_HANDLE;
        VkPipeline             pipeline    = VK_NULL_HANDLE;
        VkDescriptorPool       desc_pool   = VK_NULL_HANDLE;
        bool                   ok          = false;
    };

    auto matmul_pipeline() -> MatMulPipeline& {
        static MatMulPipeline p;
        static std::once_flag once;
        std::call_once(once, [&]() {
            if (!backend_vulkan::ensure_vulkan_ready()) return;
            auto& state = backend_vulkan::vulkan_state();

            auto bytes = backend_vulkan::load_shader_spv("matmul");
            if (bytes.empty()) return;

            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes.size();
            smci.pCode    = reinterpret_cast<const uint32_t*>(bytes.data());
            if (vkCreateShaderModule(state.device, &smci, nullptr, &p.shader)
                != VK_SUCCESS) return;

            std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
                VkDescriptorSetLayoutBinding{
                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                VkDescriptorSetLayoutBinding{
                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                VkDescriptorSetLayoutBinding{
                    2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
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
            pcr.size       = sizeof(uint32_t) * 6;

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
            pool_size.descriptorCount = 512 * 3;
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

    void matmul_impl(const TensorStorage* const* in, TensorStorage* const* out,
                     const void* ) {
        auto& state = backend_vulkan::vulkan_state();
        auto& pipe  = matmul_pipeline();
        if (!pipe.ok)
            throw std::runtime_error(vulkan_error_context(Op::MatMul, *out[0], "pipeline") +
                                     ": initialization failed");
        check_vulkan(VK_SUCCESS, Op::MatMul, *out[0], "pipeline");

        const auto& a_shape = in[0]->shape;
        const auto& b_shape = in[1]->shape;
        const auto& c_shape = out[0]->shape;
        if (a_shape.size() < 2 || b_shape.size() < 2) return;

        const std::size_t m = static_cast<std::size_t>(a_shape[a_shape.size() - 2]);
        const std::size_t k = static_cast<std::size_t>(a_shape[a_shape.size() - 1]);
        const std::size_t n = static_cast<std::size_t>(b_shape[b_shape.size() - 1]);
        if (m == 0 || n == 0 || k == 0) return;

        const std::size_t a_mat_size = m * k;
        const std::size_t b_mat_size = k * n;
        const std::size_t c_mat_size = m * n;

        std::size_t batch = 1;
        for (std::size_t i = 0; i + 2 < c_shape.size(); ++i) {
            batch *= static_cast<std::size_t>(c_shape[i]);
        }

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
        check_vulkan(desc_alloc_res, Op::MatMul, *out[0], "descriptor allocation");
        guard.descriptor_set = desc_set;

        std::array<VkDescriptorBufferInfo, 3> buffer_infos = {
            VkDescriptorBufferInfo{ buffer_of(in[0]),  in[0]->offset_bytes, VK_WHOLE_SIZE },
            VkDescriptorBufferInfo{ buffer_of(in[1]),  in[1]->offset_bytes, VK_WHOLE_SIZE },
            VkDescriptorBufferInfo{ buffer_of(out[0]), out[0]->offset_bytes, VK_WHOLE_SIZE },
        };
        std::array<VkWriteDescriptorSet, 3> writes{};
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

        struct PC {
            uint32_t M;
            uint32_t N;
            uint32_t K;
            uint32_t a_off;
            uint32_t b_off;
            uint32_t c_off;
        };

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
        check_vulkan(cmd_alloc_res, Op::MatMul, *out[0], "command allocation");
        VkCommandBuffer cmd = guard.command;

        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vulkan(vkBeginCommandBuffer(cmd, &cbbi),
                     Op::MatMul, *out[0], "command begin");

        maybe_insert_compute_barrier(cmd);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.pipe_layout, 0, 1, &desc_set,
                                0, nullptr);

        constexpr uint32_t TILE_M = 16u;
        constexpr uint32_t TILE_N = 16u;

        for (std::size_t bi = 0; bi < batch; ++bi) {
            PC pc{
                static_cast<uint32_t>(m),
                static_cast<uint32_t>(n),
                static_cast<uint32_t>(k),
                static_cast<uint32_t>(bi * a_mat_size),
                static_cast<uint32_t>(bi * b_mat_size),
                static_cast<uint32_t>(bi * c_mat_size),
            };
            vkCmdPushConstants(cmd, pipe.pipe_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            const uint32_t groups_x = (static_cast<uint32_t>(n) + TILE_N - 1u) / TILE_N;
            const uint32_t groups_y = (static_cast<uint32_t>(m) + TILE_M - 1u) / TILE_M;
            vkCmdDispatch(cmd, groups_x, groups_y, 1);
        }
        guard.finish(Op::MatMul, *out[0]);
    }

    static ::velomind::internal::KernelRegistrar _velomind_kr_matmul_f32_vk(
        DeviceType::VULKAN,
        Op::MatMul,
        ::velomind::internal::KernelDtypeKey{
            { DataType::Float32, DataType::Float32 },
            2,
            DataType::Float32
        },
        static_cast<Executable::KernelFn>(&matmul_impl));

}

}
