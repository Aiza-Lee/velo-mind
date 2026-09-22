#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "internal/log.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"

namespace velomind::internal {

using StorageCreatorFn = std::shared_ptr<velomind::TensorStorage> (*)(std::size_t bytes);
using DeviceAvailabilityFn = bool (*)() noexcept;

auto storage_creaters() -> std::array<StorageCreatorFn, MAX_DEVICE_TYPES>&;

auto register_storage_creator(velomind::DeviceType device, StorageCreatorFn fn) -> void;

auto get_storage_creator(velomind::DeviceType device) -> StorageCreatorFn;

auto register_device_availability(velomind::DeviceType device, DeviceAvailabilityFn fn) -> void;

auto is_device_available(velomind::DeviceType device) noexcept -> bool;

auto create_tensor_storage(std::size_t bytes, velomind::DeviceType device) -> std::shared_ptr<velomind::TensorStorage>;

} // namespace velomind::internal


#define VELOMIND_CONCAT_INNER_(a, b) a##b
#define VELOMIND_CONCAT_(a, b) VELOMIND_CONCAT_INNER_(a, b)

#define VELOMIND_REGISTER_STORAGE_CREATOR(DEVICE, FN)                                             \
    namespace {                                                                                   \
    [[maybe_unused]] const auto VELOMIND_CONCAT_(_velomind_auto_register_, __LINE__) = [] {       \
        ::velomind::internal::register_storage_creator(DEVICE, FN);                               \
        return true;                                                                              \
    }();                                                                                          \
    }

#define VELOMIND_REGISTER_DEVICE_AVAILABILITY(DEVICE, FN)                                         \
    namespace {                                                                                   \
    [[maybe_unused]] const auto VELOMIND_CONCAT_(_velomind_auto_register_avail_, __LINE__) = [] { \
        ::velomind::internal::register_device_availability(DEVICE, FN);                           \
        return true;                                                                              \
    }();                                                                                          \
    }
