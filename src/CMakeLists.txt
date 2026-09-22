set(VELOMIND_CORE_SRC
    tensor.cpp
    tensor_storage.cpp
    graph.cpp
    executable.cpp
    kernel.cpp
    safetensors.cpp
    dtype.cpp
    quant.cpp
    device.cpp

    internal/registry/kernel.cpp
    internal/registry/shape.cpp
    internal/registry/storage.cpp
    internal/registry/memory_transfer.cpp
)

# 后端源码直接加入同一共享库，确保静态注册器不会被链接器丢弃。
add_library(velomind_core SHARED ${VELOMIND_CORE_SRC})
add_library(velomind::velomind_core ALIAS velomind_core)
set_target_properties(velomind_core PROPERTIES
    OUTPUT_NAME "velomind-core"
    EXPORT_NAME "velomind_core"
)

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
generate_export_header(velomind_core)

target_include_directories(velomind_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/velomind>
)

target_include_directories(velomind_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(cpu)

if(VELOMIND_ENABLE_CUDA)
    target_compile_definitions(velomind_core PUBLIC VELOMIND_ENABLE_CUDA=1)
    add_subdirectory(cuda)
endif()

if(VELOMIND_ENABLE_ISPC)
    add_subdirectory(ispc)
endif()

if(VELOMIND_ENABLE_VULKAN)
    add_subdirectory(vulkan)
endif()

if(VELOMIND_ENABLE_ASAN)
    target_compile_options(velomind_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(velomind_core PUBLIC -fsanitize=address,undefined)
endif()

# 安装核心动态库并加入导出集
install(TARGETS velomind_core
    EXPORT VeloMindTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装自动生成的导出宏定义头文件
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/velomind_core_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/velomind"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/velomind_core_export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
