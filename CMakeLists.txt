cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(velo-mind VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(VELOMIND_CUDA_ARCHITECTURES "89" CACHE STRING "Target CUDA architecture(s)")

# ISPC SIMD 加速后端选项
if(NOT DEFINED VELOMIND_ENABLE_ISPC OR VELOMIND_ENABLE_ISPC)
    find_program(ISPC_EXECUTABLE ispc)
    if(ISPC_EXECUTABLE)
        set(VELOMIND_ISPC_DEFAULT ON)
    else()
        set(VELOMIND_ISPC_DEFAULT OFF)
    endif()
else()
    set(VELOMIND_ISPC_DEFAULT OFF)
endif()
option(VELOMIND_ENABLE_ISPC "Enable ispc-generated SIMD kernels" ${VELOMIND_ISPC_DEFAULT})

if(VELOMIND_ENABLE_ISPC)
    if(NOT ISPC_EXECUTABLE)
        find_program(ISPC_EXECUTABLE ispc REQUIRED)
    endif()
    if(NOT DEFINED VELOMIND_ISPC_TARGET OR VELOMIND_ISPC_TARGET STREQUAL "")
        set(VELOMIND_ISPC_TARGET "avx2-i32x8" CACHE STRING "Target ISA for ISPC compilation (e.g. avx2-i32x8, sse4-i32x4, avx512skx-x16)")
    endif()
    message(STATUS "已启用 ISPC 后端构建 (${ISPC_EXECUTABLE}, target=${VELOMIND_ISPC_TARGET})")
else()
    message(STATUS "未启用 ISPC 后端 (VELOMIND_ENABLE_ISPC=OFF)")
endif()

# CUDA 后端选项
if(NOT DEFINED VELOMIND_ENABLE_CUDA OR VELOMIND_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    set(VELOMIND_CUDA_DEFAULT ${CUDAToolkit_FOUND})
else()
    set(VELOMIND_CUDA_DEFAULT OFF)
endif()
option(VELOMIND_ENABLE_CUDA "Enable CUDA backend" ${VELOMIND_CUDA_DEFAULT})

if(VELOMIND_ENABLE_CUDA)
    if(NOT CUDAToolkit_FOUND)
        find_package(CUDAToolkit REQUIRED)
    endif()
    message(STATUS "已启用 CUDA 后端构建")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "${VELOMIND_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA 目标架构: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "未启用 CUDA 后端 (VELOMIND_ENABLE_CUDA=OFF)")
endif()

# Vulkan 计算后端选项
if(NOT DEFINED VELOMIND_ENABLE_VULKAN OR VELOMIND_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    set(VELOMIND_VULKAN_DEFAULT ${Vulkan_FOUND})
else()
    set(VELOMIND_VULKAN_DEFAULT OFF)
endif()
option(VELOMIND_ENABLE_VULKAN "Enable Vulkan compute backend" ${VELOMIND_VULKAN_DEFAULT})

if(VELOMIND_ENABLE_VULKAN)
    if(NOT Vulkan_FOUND)
        find_package(Vulkan REQUIRED)
    endif()
    find_program(GLSLANG_VALIDATOR glslangValidator)
    if(NOT GLSLANG_VALIDATOR)
        message(FATAL_ERROR "VELOMIND_ENABLE_VULKAN=ON but glslangValidator not found on PATH")
    endif()
    message(STATUS "已启用 Vulkan 后端构建 (Vulkan ${Vulkan_VERSION})")
else()
    message(STATUS "未启用 Vulkan 后端 (VELOMIND_ENABLE_VULKAN=OFF)")
endif()

option(VELOMIND_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer on VeloMind targets" OFF)
if(VELOMIND_ENABLE_ASAN)
    message(STATUS "已启用 ASan / UBSan 内存与未定义行为检测")
endif()

include(CTest)

# 拆分测试、示例与基准套件开关，遵循标准与显式控制
option(VELOMIND_BUILD_TESTS "Build VeloMind unit and integration tests" ${BUILD_TESTING})
option(VELOMIND_BUILD_EXAMPLES "Build VeloMind examples" ON)
option(VELOMIND_BUILD_BENCHMARKS "Build VeloMind benchmarks" ON)

add_subdirectory(third_party EXCLUDE_FROM_ALL SYSTEM)

add_subdirectory(src)

if(VELOMIND_BUILD_EXAMPLES OR VELOMIND_BUILD_TESTS OR VELOMIND_BUILD_BENCHMARKS)
    add_subdirectory(examples)
endif()

if(VELOMIND_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(VELOMIND_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# 安装公共头文件目录
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# 导出与生成 CMake Package 配置文件
set(VELOMIND_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/VeloMind")

install(EXPORT VeloMindTargets
    FILE VeloMindTargets.cmake
    NAMESPACE velomind::
    DESTINATION "${VELOMIND_CMAKE_DIR}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VeloMindConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VeloMindConfig.cmake"
    INSTALL_DESTINATION "${VELOMIND_CMAKE_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VeloMindConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VeloMindConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VeloMindConfigVersion.cmake"
    DESTINATION "${VELOMIND_CMAKE_DIR}"
)
