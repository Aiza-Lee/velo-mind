#pragma once

#include <cstddef>
#include "velomind_core_export.h"

namespace velomind::backend::ispc {

// 获取当前 ISPC 运行时工作线程数。
VELOMIND_CORE_EXPORT std::size_t get_thread_count();

// 设置 ISPC 运行时线程预算。传 0 时重置为环境/硬件默认值。
VELOMIND_CORE_EXPORT void set_thread_budget(std::size_t count);

} // namespace velomind::backend::ispc
