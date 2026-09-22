# 第三方依赖统一由 Git 子模块管理。

# 子模块缺失时允许本地开发环境自动补齐（按需拉取）。
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    if(VELOMIND_BUILD_TESTS AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/catch2/CMakeLists.txt")
        message(STATUS "Initializing third_party/catch2 git submodule...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive third_party/catch2
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE CATCH2_SUBMODULE_RES
        )
        if(NOT CATCH2_SUBMODULE_RES EQUAL 0)
            message(WARNING "Failed to initialize third_party/catch2 (exit code: ${CATCH2_SUBMODULE_RES})")
        endif()
    endif()
    if((VELOMIND_BUILD_TESTS OR VELOMIND_BUILD_EXAMPLES OR VELOMIND_BUILD_BENCHMARKS) AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece/CMakeLists.txt")
        message(STATUS "Initializing third_party/sentencepiece git submodule...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive third_party/sentencepiece
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            RESULT_VARIABLE SPM_SUBMODULE_RES
        )
        if(NOT SPM_SUBMODULE_RES EQUAL 0)
            message(WARNING "Failed to initialize third_party/sentencepiece (exit code: ${SPM_SUBMODULE_RES})")
        endif()
    endif()
endif()

# 屏蔽第三方库自身的编译和弃用警告。
if(MSVC)
    add_compile_options(/W0)
else()
    add_compile_options(-w)
endif()
set(CMAKE_WARN_DEPRECATED OFF)

# 测试框架（仅在启用测试构建时引入）
if(VELOMIND_BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/catch2/CMakeLists.txt")
        message(STATUS "Adding third_party/catch2...")
        add_subdirectory(catch2 EXCLUDE_FROM_ALL SYSTEM)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/catch2/extras")
        set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "third_party/catch2 not found but VELOMIND_BUILD_TESTS=ON. Run 'git submodule update --init --recursive' to fetch submodules.")
    endif()
endif()

# tokenizer 运行时（仅在需要分词器与模型的示例、测试或基准构建时引入）
if(VELOMIND_BUILD_TESTS OR VELOMIND_BUILD_EXAMPLES OR VELOMIND_BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece/CMakeLists.txt")
        message(STATUS "Adding third_party/sentencepiece...")
        set(SPM_ENABLE_SHARED ON CACHE BOOL "Build SentencePiece shared library" FORCE)
        set(SPM_BUILD_TEST OFF CACHE BOOL "Disable SentencePiece tests" FORCE)
        set(SPM_ENABLE_BENCHMARK OFF CACHE BOOL "Disable SentencePiece benchmarks" FORCE)
        set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Disable TCMalloc" FORCE)
        set(SPM_PROTOBUF_PROVIDER "module" CACHE STRING "Provider of protobuf library" FORCE)
        set(SPM_ABSL_PROVIDER "module" CACHE STRING "Provider of absl library" FORCE)
        add_subdirectory(sentencepiece EXCLUDE_FROM_ALL SYSTEM)
        target_include_directories(sentencepiece SYSTEM INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sentencepiece/src>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/sentencepiece>
        )
        if(NOT TARGET sentencepiece::sentencepiece)
            add_library(sentencepiece::sentencepiece ALIAS sentencepiece)
        endif()
    else()
        message(WARNING "third_party/sentencepiece not found. Tokenizer features will not be available.")
    endif()
endif()
