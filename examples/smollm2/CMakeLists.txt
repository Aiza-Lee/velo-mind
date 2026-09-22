set(SMOLLM2_SOURCES
    config_parser.cpp
    model_loader.cpp
    smollm2_engine.cpp
    tokenizer.cpp
)

add_library(smollm2_loader STATIC ${SMOLLM2_SOURCES})
target_include_directories(smollm2_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples/common/llama
)
target_link_libraries(smollm2_loader PUBLIC
    velomind_core
    llama_common
)
target_compile_features(smollm2_loader PUBLIC cxx_std_23)

if(VELOMIND_BUILD_EXAMPLES)
    add_executable(smollm2_example
        main.cpp
    )
    target_link_libraries(smollm2_example PRIVATE
        velomind_core
        smollm2_loader
        llama_common
    )
    target_include_directories(smollm2_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples/common/llama
    )
    target_compile_features(smollm2_example PRIVATE cxx_std_23)

    add_executable(smollm2_chat
        interactive_main.cpp
    )
    target_link_libraries(smollm2_chat PRIVATE
        smollm2_loader
        tinyllama_sample
    )
    target_include_directories(smollm2_chat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples/tinyllama
    )
    target_compile_features(smollm2_chat PRIVATE cxx_std_23)
endif()
