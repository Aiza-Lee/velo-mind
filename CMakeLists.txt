cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(velo-mind LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 作为可选后端：
#   默认按检测结果自动开启；可通过 -DVELOMIND_ENABLE_CUDA=ON/OFF 显式覆盖
find_package(CUDAToolkit)
option(VELOMIND_ENABLE_CUDA "Enable CUDA backend" ${CUDAToolkit_FOUND})

if(VELOMIND_ENABLE_CUDA)
    if(NOT CUDAToolkit_FOUND)
        # 用户显式开启但探测失败,重新以 REQUIRED 触发清晰报错
        find_package(CUDAToolkit REQUIRED)
    endif()
    message(STATUS "已启用 CUDA 后端构建")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "未启用 CUDA 后端 (VELOMIND_ENABLE_CUDA=OFF), 仅构建 CPU 部分")
endif()

include(FetchContent)

# message(STATUS "通过 FetchContent 下载 pybind11...")
# FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11.git
#     GIT_TAG        v3.0.4
# )
# FetchContent_MakeAvailable(pybind11)

message(STATUS "通过 FetchContent 下载 fmt...")
FetchContent_declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        12.2.0
)
FetchContent_MakeAvailable(fmt)
set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE CORE_SRC
    "src/*.cpp"
    "src/cpu/*.cpp"
)

if(VELOMIND_ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SRC
        "src/cuda/*.cu"
        "src/cuda/*.cpp"
    )
    list(APPEND CORE_SRC ${CUDA_SRC})
endif()

add_library(velomind_core SHARED ${CORE_SRC})
set_target_properties(velomind_core PROPERTIES OUTPUT_NAME "velomind-core")

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
generate_export_header(velomind_core)

target_include_directories(velomind_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if(VELOMIND_ENABLE_CUDA)
    target_link_libraries(velomind_core PUBLIC CUDA::cublas)
    set_target_properties(velomind_core PROPERTIES CUDA_ARCHITECTURES "native")
endif()

target_link_libraries(velomind_core PUBLIC fmt::fmt)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
