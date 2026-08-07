#include "velomind/storage.h"
#include <cstdlib>

namespace velomind {

CpuStorage::CpuStorage(size_t size_bytes) : _size_bytes(size_bytes) {
	_data_ptr = std::malloc(size_bytes);
}

CpuStorage::~CpuStorage() {
	if (_data_ptr) {
		std::free(_data_ptr);
	}
}

} // namespace velomind
