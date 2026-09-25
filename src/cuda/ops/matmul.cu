#include "velomind/executable.h"
#include "velomind/ops.h"
#include "velomind/tensor_storage.h"
#include "velomind/types.h"
#include "velomind/dtype.h"

#include <cstddef>
#include <stdexcept>
#include <cublas_v2.h>

#include "backend_error.h"
#include "internal/registry/kernel.h"
#include "internal/shape_helpers.h"

namespace velomind::backend::cuda {

namespace {

    void check_cublas(cublasStatus_t status, const TensorStorage& output,
                      const char* stage) {
        const auto context = cuda_error_context(Op::MatMul, output, stage);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(context + ": cuBLAS status " +
                                     std::to_string(static_cast<int>(status)));
        if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
            const auto key = std::string("CUDA:MatMul:") + stage;
            if (key == injected) throw std::runtime_error(context + ": injected failure");
        }
    }

    template <typename T>
    struct CublasType;
    template <> struct CublasType<float>      { static constexpr cudaDataType_t value = CUDA_R_32F; };
    template <> struct CublasType<float16_t>  { static constexpr cudaDataType_t value = CUDA_R_16F; };
    template <> struct CublasType<bfloat16_t> { static constexpr cudaDataType_t value = CUDA_R_16BF; };

    template <typename T1, typename T2, typename T3>
    void matmul_cublas_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
        const T1* a_data = static_cast<const T1*>(in[0]->data);
        const T2* b_data = static_cast<const T2*>(in[1]->data);
        T3*       c_data = static_cast<T3*>(out[0]->data);

        const auto& a_shape = in[0]->shape;
        const auto& b_shape = in[1]->shape;
        const auto& c_shape = out[0]->shape;

        const std::size_t a_rank = a_shape.size();
        const std::size_t b_rank = b_shape.size();
        const std::size_t c_rank = c_shape.size();

        if (a_rank < 2 || b_rank < 2 || c_rank < 2) {
            throw std::runtime_error("velomind::cuda::MatMul: rank must be >= 2");
        }

        const std::size_t M = static_cast<std::size_t>(a_shape[a_rank - 2]);
        const std::size_t K = static_cast<std::size_t>(a_shape[a_rank - 1]);
        const std::size_t N = static_cast<std::size_t>(b_shape[b_rank - 1]);

        if (static_cast<std::size_t>(b_shape[b_rank - 2]) != K) {
            throw std::runtime_error("velomind::cuda::MatMul: K dimension mismatch");
        }

        std::size_t batch = 1;
        for (std::size_t i = 0; i + 2 < c_rank; ++i) {
            batch *= static_cast<std::size_t>(c_shape[i]);
        }

        const std::size_t a_mat_size = M * K;
        const std::size_t b_mat_size = (b_rank >= 3) ? (K * N) : 0;
        const std::size_t c_mat_size = M * N;

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        const auto b_strides = in[1]->effective_strides();
        const bool b_transposed = (b_rank >= 2 &&
                                   b_strides[b_rank - 2] == 1 &&
                                   b_strides[b_rank - 1] == static_cast<dim_t>(K));
        const cublasOperation_t op_b = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int ldb = b_transposed ? static_cast<int>(K) : static_cast<int>(N);

        auto& ctx = get_cuda_context();
        auto handle = ctx.cublas();

        // cuBLAS 按列主序解释内存，因此交换 A/B，以 Cᵀ=Bᵀ×Aᵀ 计算行主序结果。
        if (batch == 1) {
            check_cublas(cublasGemmEx(handle,
                         op_b, CUBLAS_OP_N,
                         static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                         &alpha,
                         b_data, CublasType<T2>::value, ldb,
                         a_data, CublasType<T1>::value, static_cast<int>(K),
                         &beta,
                         c_data, CublasType<T3>::value, static_cast<int>(N),
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT), *out[0], "gemm");
        } else if (b_mat_size == 0 && in[0]->is_contiguous()) {
            // 二维广播权重且输入连续时折叠为单次大 GEMM
            const std::size_t total_m = batch * M;
            check_cublas(cublasGemmEx(handle,
                         op_b, CUBLAS_OP_N,
                         static_cast<int>(N), static_cast<int>(total_m), static_cast<int>(K),
                         &alpha,
                         b_data, CublasType<T2>::value, ldb,
                         a_data, CublasType<T1>::value, static_cast<int>(K),
                         &beta,
                         c_data, CublasType<T3>::value, static_cast<int>(N),
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT), *out[0], "gemm");
        } else {
            // 批量 GEMM（如多头注意力），通过 Strided Batched GEMM 单次派发
            const auto stride_b = static_cast<long long int>(b_mat_size);
            const auto stride_a = static_cast<long long int>(a_mat_size);
            const auto stride_c = static_cast<long long int>(c_mat_size);

            check_cublas(cublasGemmStridedBatchedEx(handle,
                         op_b, CUBLAS_OP_N,
                         static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                         &alpha,
                         b_data, CublasType<T2>::value, ldb, stride_b,
                         a_data, CublasType<T1>::value, static_cast<int>(K), stride_a,
                         &beta,
                         c_data, CublasType<T3>::value, static_cast<int>(N), stride_c,
                         static_cast<int>(batch),
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT), *out[0], "gemm");
        }

        if (const char* injected = std::getenv("VELOMIND_FAIL_BACKEND")) {
            const auto key = std::string("CUDA:MatMul:sync");
            if (key == injected) check_cuda(cudaSuccess, Op::MatMul, *out[0], "sync");
        }

        if (is_cuda_sync_debug_enabled()) {
            check_cuda(cudaStreamSynchronize(ctx.stream_handle()), Op::MatMul, *out[0], "sync");
        }
    }

    template <typename T1, typename T2, typename T3>
    __global__ void matmul_mixed_kernel(const T1* __restrict__ a,
                                        const T2* __restrict__ b,
                                        T3* __restrict__ c,
                                        std::size_t M, std::size_t K, std::size_t N,
                                        std::size_t batch,
                                        std::size_t a_stride_batch,
                                        std::size_t b_stride_batch,
                                        std::size_t c_stride_batch,
                                        bool b_transposed) {
        std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        std::size_t m = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        std::size_t b_idx = blockIdx.z;

        if (m >= M || n >= N || b_idx >= batch) return;

        const T1* a_mat = a + b_idx * a_stride_batch + m * K;
        const T2* b_mat = b + b_idx * b_stride_batch;
        T3*       c_mat = c + b_idx * c_stride_batch + m * N;

        float acc = 0.0f;
        if (b_transposed) {
            const T2* b_row = b_mat + n * K;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<float>(a_mat[k]) * static_cast<float>(b_row[k]);
            }
        } else {
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<float>(a_mat[k]) * static_cast<float>(b_mat[k * N + n]);
            }
        }
        c_mat[n] = static_cast<T3>(acc);
    }

    template <typename T1, typename T2, typename T3>
    void matmul_mixed_impl(const TensorStorage* const* in, TensorStorage* const* out, const void* ) {
        const T1* a_data = static_cast<const T1*>(in[0]->data);
        const T2* b_data = static_cast<const T2*>(in[1]->data);
        T3*       c_data = static_cast<T3*>(out[0]->data);

        const auto& a_shape = in[0]->shape;
        const auto& b_shape = in[1]->shape;
        const auto& c_shape = out[0]->shape;

        const std::size_t a_rank = a_shape.size();
        const std::size_t b_rank = b_shape.size();
        const std::size_t c_rank = c_shape.size();

        if (a_rank < 2 || b_rank < 2 || c_rank < 2) {
            throw std::runtime_error("velomind::cuda::MatMul: rank must be >= 2");
        }

        const std::size_t M = static_cast<std::size_t>(a_shape[a_rank - 2]);
        const std::size_t K = static_cast<std::size_t>(a_shape[a_rank - 1]);
        const std::size_t N = static_cast<std::size_t>(b_shape[b_rank - 1]);

        if (static_cast<std::size_t>(b_shape[b_rank - 2]) != K) {
            throw std::runtime_error("velomind::cuda::MatMul: K dimension mismatch");
        }

        std::size_t batch = 1;
        for (std::size_t i = 0; i + 2 < c_rank; ++i) {
            batch *= static_cast<std::size_t>(c_shape[i]);
        }

        const std::size_t a_stride_batch = M * K;
        const std::size_t b_stride_batch = (b_rank >= 3) ? (K * N) : 0;
        const std::size_t c_stride_batch = M * N;

        const auto b_strides = in[1]->effective_strides();
        const bool b_transposed = (b_rank >= 2 &&
                                   b_strides[b_rank - 2] == 1 &&
                                   b_strides[b_rank - 1] == static_cast<dim_t>(K));

        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned int>((N + 15) / 16),
                  static_cast<unsigned int>((M + 15) / 16),
                  static_cast<unsigned int>(batch));

        auto stream = get_cuda_context().stream_handle();
        matmul_mixed_kernel<<<grid, block, 0, stream>>>(
            a_data, b_data, c_data, M, K, N, batch,
            a_stride_batch, b_stride_batch, c_stride_batch, b_transposed);
        check_cuda_kernel(Op::MatMul, *out[0]);
    }

    // 同精度 cuBLAS 内核
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float32,  DataType::Float32,  DataType::Float32,  matmul_cublas_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float16,  DataType::Float16,  DataType::Float16,  matmul_cublas_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float16,  DataType::Float16,  DataType::Float32,  matmul_cublas_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::BFloat16, DataType::BFloat16, DataType::BFloat16, matmul_cublas_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::BFloat16, DataType::BFloat16, DataType::Float32,  matmul_cublas_impl);

    // 混合精度自定义内核（FP32 累加）
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float32,  DataType::Float16,  DataType::Float32,  matmul_mixed_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float32,  DataType::BFloat16, DataType::Float32,  matmul_mixed_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::Float16,  DataType::Float32,  DataType::Float32,  matmul_mixed_impl);
    VELOMIND_REGISTER_BINARY_OP(DeviceType::CUDA, Op::MatMul, DataType::BFloat16, DataType::Float32,  DataType::Float32,  matmul_mixed_impl);

} // namespace

} // namespace velomind::backend::cuda
