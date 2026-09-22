#include "internal/registry/memory_transfer.h"

namespace velomind::internal {

auto memory_transfers() -> std::array<MemoryTransfer, MAX_DEVICE_TYPES>& {
    static std::array<MemoryTransfer, MAX_DEVICE_TYPES> r{};
    return r;
}

void register_memory_transfer(DeviceType device, MemoryTransfer t) {
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) return;
    auto& slot = memory_transfers()[idx];
    if (slot.copy_h2d != nullptr && t.copy_h2d != nullptr) {
        VELOMIND_LOG(Warn,
            "device {} 已注册的内存传输方法 (host->device) 被覆盖",
            static_cast<int>(device));
    }
    if (slot.copy_d2h != nullptr && t.copy_d2h != nullptr) {
        VELOMIND_LOG(Warn,
            "device {} 已注册的内存传输方法 (device->host) 被覆盖",
            static_cast<int>(device));
    }
    slot = t;
}

auto get_memory_transfer(DeviceType device) -> MemoryTransfer {
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) return MemoryTransfer{};
    return memory_transfers()[idx];
}

} // namespace velomind::internal
