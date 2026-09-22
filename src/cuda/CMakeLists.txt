target_sources(velomind_core PRIVATE
    cuda_context.cu
    storage.cu

    ops/add.cu
    ops/sub.cu
    ops/mul.cu
    ops/div.cu
    ops/neg.cu
    ops/abs.cu
    ops/relu.cu
    ops/rsqrt.cu
    ops/sigmoid.cu
    ops/tanh.cu
    ops/softmax.cu
    ops/matmul.cu
    ops/embedding.cu
    ops/rmsnorm.cu
    ops/rotary_embedding.cu
    ops/reshape.cu
    ops/transpose.cu
    ops/concat.cu
    ops/repeat_kv.cu
    ops/slice.cu
    ops/fused_attention.cu
    ops/quantized_matmul.cu
)

target_link_libraries(velomind_core PUBLIC CUDA::cublas)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(velomind_core PROPERTIES
        CUDA_ARCHITECTURES "native"
    )
endif()
