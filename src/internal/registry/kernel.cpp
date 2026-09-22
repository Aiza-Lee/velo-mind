#include "internal/registry/kernel.h"

#include <stdexcept>

namespace velomind::internal {

auto operator==(const KernelDtypeKey& a, const KernelDtypeKey& b) noexcept -> bool {
    if (a.num_inputs != b.num_inputs) return false;
    if (a.output_dtype != b.output_dtype) return false;
    for (std::size_t i = 0; i < a.num_inputs; ++i) {
        if (a.input_dtypes[i] != b.input_dtypes[i]) return false;
    }
    return true;
}

using OpTableSlot = std::array<std::vector<KernelEntry>, MAX_OPS>;

auto op_kernels(DeviceType device) -> OpTableSlot& {
    static OpTableSlot op_table[MAX_DEVICE_TYPES]{};
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) {
        throw std::out_of_range(
            "op_kernels: device index out of range (DeviceType value "
            "exceeds MAX_DEVICE_TYPES — enum was extended without "
            "bumping the table size)");
    }
    return op_table[idx];
}

auto register_op_kernel(DeviceType           device,
                        Op                   op,
                        KernelDtypeKey       k,
                        Executable::KernelFn fn) -> void
{
    op_kernels(device)[static_cast<std::size_t>(op)].push_back({k, fn});
}

KernelRegistrar::KernelRegistrar(DeviceType           device,
                                 Op                   op,
                                 KernelDtypeKey       k,
                                 Executable::KernelFn fn)
{
    register_op_kernel(device, op, k, fn);
}

} // namespace velomind::internal
