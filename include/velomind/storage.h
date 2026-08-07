#include "velomind_core_export.h"
#include "velomind/types.h"

namespace velomind {


class VELOMIND_CORE_EXPORT IStorage {
public:
	virtual ~IStorage() = default;
	virtual auto data() -> void* = 0;
	virtual auto data() const -> const void* = 0;
	virtual auto device_type() const -> DeviceType = 0;
};

class VELOMIND_CORE_EXPORT CpuStorage : public IStorage {
public:
	CpuStorage(size_t size_bytes);
	~CpuStorage() override;

	auto data() -> void* override { return _data_ptr; }
	auto data() const -> const void* override { return _data_ptr; }
	auto device_type() const -> DeviceType override { return DeviceType::CPU; }
	
private:
	void* _data_ptr;
	size_t _size_bytes;
};

class VELOMIND_CORE_EXPORT CudaStorage : public IStorage {
public:
	CudaStorage(size_t size_bytes);
	~CudaStorage() override;

	auto data() -> void* override { return _data_ptr; }
	auto data() const -> const void* override { return _data_ptr; }
	auto device_type() const -> DeviceType override { return DeviceType::CUDA; }
private:
	void* _data_ptr;
	size_t _size_bytes;
};


} // namespace velomind
