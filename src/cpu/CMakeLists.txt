target_sources(velomind_core PRIVATE
    storage.cpp

    ops/add.cpp
    ops/sub.cpp
    ops/mul.cpp
    ops/div.cpp
    ops/neg.cpp
    ops/abs.cpp
    ops/relu.cpp
    ops/sigmoid.cpp
    ops/tanh.cpp
    ops/softmax.cpp
    ops/matmul.cpp
    ops/rsqrt.cpp
    ops/rmsnorm.cpp
    ops/embedding.cpp
    ops/rotary_embedding.cpp
    ops/reshape.cpp
    ops/transpose.cpp
    ops/concat.cpp
    ops/repeat_kv.cpp
    ops/slice.cpp
    ops/fused_attention.cpp
    ops/quantized_matmul.cpp
)
