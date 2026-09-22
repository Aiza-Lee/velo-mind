add_library(llama_common STATIC
    llama_rope_mask.cpp
    llama_graph.cpp
    llama_engine.cpp
)
target_link_libraries(llama_common PUBLIC
    velomind_core
)
target_include_directories(llama_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(llama_common PUBLIC cxx_std_23)
