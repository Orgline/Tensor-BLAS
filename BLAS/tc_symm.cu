#include "../include/TensorBLAS.h" 

void tc_symm(cublasHandle_t handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc, __half* work)
{
    dim3 grida((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    copy_lower_to_upper<<<grida, block>>>(m, A, m);
    
    __half* Ah = work;
    __half* Bh = work + m*m;
    
    s2h<<<grida, block>>>(m, m, A, lda, Ah, lda);
    dim3 gridb((m+31)/32, (n+31)/32);
    s2h<<<grida, block>>>(m, n, B, ldb, Bh, ldb);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                &alpha, Ah, CUDA_R_16F, lda, Bh, CUDA_R_16F, ldb,
                &beta, C, CUDA_R_32F, ldc, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}