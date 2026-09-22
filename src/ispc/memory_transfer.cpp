#include "velomind/types.h"

#include <cstring>

#include "internal/registry/memory_transfer.h"

namespace velomind::internal {

void ispc_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void ispc_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
    std::memcpy(dst, src, bytes);
}

VELOMIND_REGISTER_MEMORY_TRANSFER(DeviceType::ISPC, ispc_memcpy_h2d, ispc_memcpy_d2h)

}
