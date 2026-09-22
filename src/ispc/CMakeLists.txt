# 支持配置 ISPC 目标指令集（默认 avx2-i32x8，可设为 avx512skx-x16、sse4-i32x4 等）。
if(NOT DEFINED VELOMIND_ISPC_TARGET OR VELOMIND_ISPC_TARGET STREQUAL "")
    set(VELOMIND_ISPC_TARGET "avx2-i32x8" CACHE STRING "Target ISA for ISPC compilation")
endif()
set(ISPC_TARGET "${VELOMIND_ISPC_TARGET}")
set(ISPC_SRC_DIR     "${CMAKE_CURRENT_SOURCE_DIR}")
set(ISPC_OPS_DIR     "${ISPC_SRC_DIR}/ops")
set(ISPC_OUT_DIR     "${CMAKE_BINARY_DIR}/ispc_gen")
set(ISPC_INCLUDE_DIR "${ISPC_OUT_DIR}")

target_sources(velomind_core PRIVATE
    storage.cpp
    memory_transfer.cpp
    task_system.cpp

    ops/elementwise_add.cpp
    ops/elementwise_sub.cpp
    ops/elementwise_mul.cpp
    ops/elementwise_div.cpp
    ops/elementwise_neg.cpp
    ops/elementwise_abs.cpp
    ops/elementwise_relu.cpp
    ops/elementwise_sigmoid.cpp
    ops/elementwise_tanh.cpp
    ops/elementwise_rsqrt.cpp

    ops/matmul.cpp
    ops/rmsnorm.cpp
    ops/softmax.cpp
    ops/rotary_embedding.cpp
    ops/embedding.cpp
    ops/transpose.cpp
    ops/reshape.cpp
    ops/concat.cpp
    ops/repeat_kv.cpp
    ops/slice.cpp
)

set(ISPC_KERNELS
    elementwise_add
    elementwise_sub
    elementwise_mul
    elementwise_div
    elementwise_neg
    elementwise_abs
    elementwise_relu
    elementwise_sigmoid
    elementwise_tanh
    elementwise_rsqrt

    matmul
    rmsnorm
    softmax
    rotary_embedding
    embedding
    transpose
)

set(ISPC_GEN_OBJS)
foreach(kernel IN LISTS ISPC_KERNELS)
    set(obj "${ISPC_OUT_DIR}/${kernel}_${ISPC_TARGET}.o")
    list(APPEND ISPC_GEN_OBJS "${obj}")
    add_custom_command(
        OUTPUT  "${obj}"
        COMMAND ${ISPC_EXECUTABLE}
                --target=${ISPC_TARGET}
                -O3
                --pic
                --wno-perf
                -o "${obj}"
                "${ISPC_OPS_DIR}/${kernel}.ispc"
        DEPENDS "${ISPC_OPS_DIR}/${kernel}.ispc"
        COMMENT "Compiling ispc kernel: ${kernel}.ispc (target=${ISPC_TARGET})"
        VERBATIM)
endforeach()

file(MAKE_DIRECTORY "${ISPC_OUT_DIR}")

target_sources(velomind_core PRIVATE ${ISPC_GEN_OBJS})

add_custom_target(velomind_ispc_objects DEPENDS ${ISPC_GEN_OBJS})
add_dependencies(velomind_core velomind_ispc_objects)

target_include_directories(velomind_core PRIVATE "${ISPC_INCLUDE_DIR}")
