#include "internal/registry/storage.h"

namespace velomind::internal {

auto storage_creaters() -> std::array<StorageCreatorFn, MAX_DEVICE_TYPES>& {
    static std::array<StorageCreatorFn, MAX_DEVICE_TYPES> r{};
    return r;
}

void register_storage_creator(velomind::DeviceType device, StorageCreatorFn fn) {
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) return;
    auto& slot = storage_creaters()[idx];
    if (slot != nullptr) {
        VELOMIND_LOG(Warn,
            "storage allocator for device {} is being overwritten "
            "(a prior allocator is already registered for this slot)",
            static_cast<int>(device));
    }
    slot = fn;
}

auto get_storage_creator(velomind::DeviceType device) -> StorageCreatorFn {
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) return nullptr;
    return storage_creaters()[idx];
}

namespace {
    auto availability_checkers() -> std::array<DeviceAvailabilityFn, MAX_DEVICE_TYPES>& {
        static std::array<DeviceAvailabilityFn, MAX_DEVICE_TYPES> r{};
        return r;
    }
}

void register_device_availability(velomind::DeviceType device, DeviceAvailabilityFn fn) {
    auto idx = static_cast<std::size_t>(device);
    if (idx < MAX_DEVICE_TYPES) {
        availability_checkers()[idx] = fn;
    }
}

auto is_device_available(velomind::DeviceType device) noexcept -> bool {
    auto idx = static_cast<std::size_t>(device);
    if (idx >= MAX_DEVICE_TYPES) return false;
    auto fn = availability_checkers()[idx];
    if (fn != nullptr) {
        return fn();
    }
    if (device == DeviceType::CPU) return true;
#ifdef VELOMIND_ENABLE_ISPC
    if (device == DeviceType::ISPC) return get_storage_creator(device) != nullptr;
#endif
    return get_storage_creator(device) != nullptr;
}

} // namespace velomind::internal
