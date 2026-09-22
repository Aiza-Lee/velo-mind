#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstdlib>
#include <cstring>

#include "internal/registry/memory_transfer.h"
#include "internal/registry/storage.h"

namespace velomind::backend::cpu {

namespace {

    auto cpu_allocator(std::size_t bytes) -> std::shared_ptr<TensorStorage> {
        void* buf = bytes > 0 ? std::calloc(1, bytes) : nullptr;
        auto s = std::shared_ptr<TensorStorage>(
            new TensorStorage{},
            [buf](TensorStorage* p) {
                std::free(buf);
                delete p;
            });
        s->size_bytes     = bytes;
        s->capacity_bytes = bytes;
        s->offset_bytes   = 0;
        s->device         = DeviceType::CPU;
        s->data           = buf;
        return s;
    }

    void cpu_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
        std::memcpy(dst, src, bytes);
    }

    void cpu_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
        std::memcpy(dst, src, bytes);
    }

    VELOMIND_REGISTER_STORAGE_CREATOR(DeviceType::CPU, cpu_allocator)
    VELOMIND_REGISTER_MEMORY_TRANSFER(DeviceType::CPU, cpu_memcpy_h2d, cpu_memcpy_d2h)

}

}
