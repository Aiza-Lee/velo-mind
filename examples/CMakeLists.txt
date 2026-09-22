cmake_minimum_required(VERSION 3.28)
project(velomind_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(VELOMIND_BUILD_EXAMPLES)
    add_executable(example_basic example_basic.cpp)
    target_link_libraries(example_basic PRIVATE velomind_core)

    add_executable(example_mlp example_mlp.cpp)
    target_link_libraries(example_mlp PRIVATE velomind_core)

    add_executable(example_mnist example_mnist.cpp)
    target_link_libraries(example_mnist PRIVATE velomind_core)
endif()

add_subdirectory(common/llama)
add_subdirectory(tinyllama)
add_subdirectory(smollm2)

include(CTest)
if(BUILD_TESTING AND VELOMIND_BUILD_EXAMPLES)
    add_test(NAME example_basic     COMMAND example_basic)
    add_test(NAME example_mlp       COMMAND example_mlp)
    add_test(NAME example_mnist     COMMAND example_mnist)
    add_test(NAME tinyllama_example COMMAND tinyllama_example)
    add_test(NAME smollm2_example   COMMAND smollm2_example)
endif()
