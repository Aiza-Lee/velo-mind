set(VULKAN_OPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ops")
set(VULKAN_OUT_DIR "${CMAKE_BINARY_DIR}/vulkan_gen")

target_sources(velomind_core PRIVATE
    storage.cpp
    init.cpp
    shader_loader.cpp

    ops/elementwise_add.cpp
    ops/elementwise_mul.cpp
    ops/embedding.cpp
    ops/matmul.cpp
    ops/rmsnorm.cpp
    ops/rotary_embedding.cpp
    ops/sigmoid.cpp
    ops/softmax.cpp

    ops/reshape.cpp
    ops/transpose.cpp
    ops/repeat_kv.cpp
    ops/slice.cpp
    ops/concat.cpp
)

set(VULKAN_SHADERS
    elementwise_add
    elementwise_mul
    embedding
    matmul
    rmsnorm
    rotary_embedding
    sigmoid
    softmax
    transpose
)

set(VULKAN_GEN_OBJS)
foreach(shader IN LISTS VULKAN_SHADERS)
    set(spv "${VULKAN_OUT_DIR}/${shader}.spv")
    list(APPEND VULKAN_GEN_OBJS "${spv}")
    add_custom_command(
        OUTPUT  "${spv}"
        COMMAND ${GLSLANG_VALIDATOR}
                -V
                -S comp
                --target-env spirv1.4
                -o "${spv}"
                "${VULKAN_OPS_DIR}/${shader}.comp"
        DEPENDS "${VULKAN_OPS_DIR}/${shader}.comp"
        COMMENT "Compiling Vulkan shader: ${shader}.comp -> ${shader}.spv"
        VERBATIM)
endforeach()

file(MAKE_DIRECTORY "${VULKAN_OUT_DIR}")

set(VULKAN_SHADER_PATHS_HEADER "${VULKAN_OUT_DIR}/vulkan_shader_paths.h")
file(WRITE "${VULKAN_SHADER_PATHS_HEADER}"
     "// 由 src/vulkan/CMakeLists.txt 自动生成：SPIR-V 文件路径。\n"
     "#pragma once\n\n")
foreach(shader IN LISTS VULKAN_SHADERS)
    file(APPEND "${VULKAN_SHADER_PATHS_HEADER}"
         "#define VELOMIND_VULKAN_SPV_PATH_${shader} \"${VULKAN_OUT_DIR}/${shader}.spv\"\n")
endforeach()

add_custom_target(velomind_vulkan_shaders DEPENDS ${VULKAN_GEN_OBJS})
add_dependencies(velomind_core velomind_vulkan_shaders)

target_link_libraries(velomind_core PUBLIC Vulkan::Vulkan)
target_include_directories(velomind_core PRIVATE "${VULKAN_OUT_DIR}")

# 传递部署期与构建期着色器默认查找目录
include(GNUInstallDirs)
set(VELOMIND_VULKAN_SHADER_INSTALL_DIR "${CMAKE_INSTALL_FULL_DATADIR}/velomind/shaders")

target_compile_definitions(velomind_core PRIVATE
    VELOMIND_VULKAN_SHADER_INSTALL_DIR="${VELOMIND_VULKAN_SHADER_INSTALL_DIR}"
    VELOMIND_VULKAN_SHADER_BUILD_DIR="${VULKAN_OUT_DIR}"
)

# 安装 SPIR-V 着色器至系统数据目录
install(FILES ${VULKAN_GEN_OBJS}
    DESTINATION "${CMAKE_INSTALL_DATADIR}/velomind/shaders"
)
