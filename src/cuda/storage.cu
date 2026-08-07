#include "velomind/storage.h"
#include <cstddef>
#include <cstdio>

namespace velomind {

CudaStorage::CudaStorage(size_t size_bytes) : _size_bytes(size_bytes) {
	auto err = cudaMalloc(&_data_ptr, size_bytes);
}

CudaStorage::~CudaStorage() {
	if (_data_ptr) {
		cudaFree(_data_ptr);
	}
}

} // namespace velomind
