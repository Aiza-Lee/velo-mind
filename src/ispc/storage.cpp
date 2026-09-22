#include "velomind/tensor_storage.h"
#include "velomind/types.h"

#include <cstdlib>

#include "internal/registry/storage.h"

namespace velomind::backend::ispc {

namespace {

    auto ispc_allocator(std::size_t bytes) -> std::shared_ptr<TensorStorage> {

        void* buf = nullptr;
        if (bytes > 0) {
            // 64 字节对齐，满足 AVX2/AVX-512 SIMD 向量访问与高速缓存行边界对齐。
            if (posix_memalign(&buf, 64, bytes) != 0) {
                buf = nullptr;
            }
        }
        auto s = std::shared_ptr<TensorStorage>(
            new TensorStorage{},
            [buf](TensorStorage* p) {
                std::free(buf);
                delete p;
            });
        s->size_bytes     = bytes;
        s->capacity_bytes = bytes;
        s->offset_bytes   = 0;
        s->device         = DeviceType::ISPC;
        s->data           = buf;
        return s;
    }

    VELOMIND_REGISTER_STORAGE_CREATOR(DeviceType::ISPC, ispc_allocator)

}

}
