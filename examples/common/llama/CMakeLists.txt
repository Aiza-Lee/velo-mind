add_library(llama_common STATIC
    llama_rope_mask.cpp
    llama_graph.cpp
    llama_engine.cpp
    llama_model_finder.cpp
    llama_interactive_console.cpp
)
target_compile_definitions(llama_common PRIVATE
    VELOMIND_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)
target_link_libraries(llama_common PUBLIC
    velomind_core
)
target_include_directories(llama_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(llama_common PUBLIC cxx_std_23)
