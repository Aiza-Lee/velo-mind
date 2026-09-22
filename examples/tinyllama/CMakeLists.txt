add_library(tinyllama_tokenizer STATIC
    tokenizer.cpp
)
target_link_libraries(tinyllama_tokenizer PUBLIC
    sentencepiece
)
target_include_directories(tinyllama_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples/common/llama
)

add_library(tinyllama_sample STATIC
    sample.cpp
)
target_link_libraries(tinyllama_sample PUBLIC
    velomind_core
)
target_include_directories(tinyllama_sample PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(tinyllama_model STATIC
    graph_builder.cpp
    model_loader.cpp
    tinyllama_engine.cpp
)
target_link_libraries(tinyllama_model PUBLIC
    velomind_core
    llama_common
    tinyllama_tokenizer
    tinyllama_sample
)
target_include_directories(tinyllama_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples/common/llama
)

if(VELOMIND_BUILD_EXAMPLES)
    add_executable(tinyllama_example
        main.cpp
    )
    target_link_libraries(tinyllama_example PRIVATE
        velomind_core
        tinyllama_model
    )
    target_include_directories(tinyllama_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples/common/llama
    )

    add_executable(tinyllama_chat
        interactive_main.cpp
    )
    target_link_libraries(tinyllama_chat PRIVATE
        tinyllama_model
    )
    target_include_directories(tinyllama_chat PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
