set(VELOMIND_TESTS_SRC
    test_basic.cpp
    test_softmax.cpp
    test_validation.cpp
    test_sample.cpp
    test_interactive.cpp
    cpu/test_cpu.cpp
    cpu/test_cpu_matmul.cpp
    io/test_safetensors.cpp
    io/test_tokenizer.cpp
    io/test_smollm2_tokenizer.cpp
    test_tinyllama_e2e.cpp
    test_smollm2_e2e.cpp
    test_coverage_matrix.cpp
    test_dtype_fp16_bf16.cpp
    test_fused_attention.cpp
    test_quantization.cpp
)

if(VELOMIND_ENABLE_CUDA)
    list(APPEND VELOMIND_TESTS_SRC cuda/test_cuda.cpp)
endif()

if(VELOMIND_ENABLE_ISPC)
    list(APPEND VELOMIND_TESTS_SRC ispc/test_ispc.cpp)
endif()

if(VELOMIND_ENABLE_VULKAN)
    list(APPEND VELOMIND_TESTS_SRC vulkan/test_vulkan.cpp)
endif()

add_executable(velomind_tests ${VELOMIND_TESTS_SRC})
target_link_libraries(velomind_tests PRIVATE
    velomind_core
    tinyllama_model
    tinyllama_tokenizer
    tinyllama_sample
    smollm2_loader
    llama_common
    Catch2::Catch2WithMain
)
target_include_directories(velomind_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/examples/tinyllama
    ${CMAKE_SOURCE_DIR}/examples/smollm2
    ${CMAKE_SOURCE_DIR}/examples/common/llama
)

target_compile_definitions(velomind_tests PRIVATE
    VELOMIND_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

if(VELOMIND_ENABLE_CUDA)
    target_compile_definitions(velomind_tests PRIVATE VELOMIND_ENABLE_CUDA=1)
endif()

if(VELOMIND_ENABLE_ISPC)
    target_compile_definitions(velomind_tests PRIVATE VELOMIND_ENABLE_ISPC=1)
endif()

if(VELOMIND_ENABLE_VULKAN)
    target_compile_definitions(velomind_tests PRIVATE VELOMIND_ENABLE_VULKAN=1)
endif()

if(VELOMIND_ENABLE_ASAN)
    target_compile_options(velomind_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(velomind_tests PRIVATE -fsanitize=address,undefined)
endif()

include(Catch)
catch_discover_tests(velomind_tests)

# 别名与快捷运行目标，支持不同构建习惯与拼写
add_custom_target(velomind-test DEPENDS velomind_tests)
add_custom_target(velmind-test DEPENDS velomind_tests)
add_custom_target(velomind_test DEPENDS velomind_tests)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS velomind_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "运行测试套件"
)
