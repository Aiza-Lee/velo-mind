#pragma once

#include <array>
#include <cstddef>

#include "internal/log.h"
#include "velomind/types.h"

namespace velomind::internal {

struct MemoryTransfer {
    void (*copy_h2d)(void* dst, const void* src, std::size_t bytes) = nullptr;
    void (*copy_d2h)(void* dst, const void* src, std::size_t bytes) = nullptr;
};

auto memory_transfers() -> std::array<MemoryTransfer, MAX_DEVICE_TYPES>&;

void register_memory_transfer(DeviceType device, MemoryTransfer t);

auto get_memory_transfer(DeviceType device) -> MemoryTransfer;

} // namespace velomind::internal


#define VELOMIND_CONCAT_INNER__(a, b) a##b
#define VELOMIND_CONCAT__(a, b) VELOMIND_CONCAT_INNER__(a, b)

#define VELOMIND_REGISTER_MEMORY_TRANSFER(DEVICE, H2D_FN, D2H_FN)                           \
    namespace {                                                                             \
    [[maybe_unused]] const auto VELOMIND_CONCAT__(_velomind_mt_register_, __LINE__) = [] {  \
        ::velomind::internal::register_memory_transfer(                                     \
            (DEVICE),                                                                       \
            ::velomind::internal::MemoryTransfer{                                           \
                (H2D_FN),                                                                   \
                (D2H_FN),                                                                   \
            });                                                                             \
        return true;                                                                        \
    }();                                                                                    \
    }
