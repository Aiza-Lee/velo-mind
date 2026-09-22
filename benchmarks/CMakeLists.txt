add_executable(benchmark_baseline
    benchmark_baseline.cpp
)

target_link_libraries(benchmark_baseline PRIVATE
    velomind_core
    smollm2_loader
    llama_common
)

target_include_directories(benchmark_baseline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/examples/common/llama
    ${CMAKE_SOURCE_DIR}/examples/smollm2
    ${CMAKE_SOURCE_DIR}/examples/tinyllama
)

if(VELOMIND_ENABLE_CUDA)
    target_compile_definitions(benchmark_baseline PRIVATE VELOMIND_ENABLE_CUDA=1)
    target_link_libraries(benchmark_baseline PRIVATE CUDA::cudart)
endif()

if(VELOMIND_ENABLE_ISPC)
    target_compile_definitions(benchmark_baseline PRIVATE VELOMIND_ENABLE_ISPC=1)
endif()

if(VELOMIND_ENABLE_VULKAN)
    target_compile_definitions(benchmark_baseline PRIVATE VELOMIND_ENABLE_VULKAN=1)
endif()

if(VELOMIND_ENABLE_ASAN)
    target_compile_options(benchmark_baseline PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(benchmark_baseline PRIVATE -fsanitize=address,undefined)
endif()

# 快速冒烟测试：验证基准评测工具在合成模式下能正常解析并执行
add_test(NAME benchmark_baseline_smoke
    COMMAND benchmark_baseline --device cpu --seq-lens 1,8 --iters 1 --warmup 0 --synthetic --quiet
)
