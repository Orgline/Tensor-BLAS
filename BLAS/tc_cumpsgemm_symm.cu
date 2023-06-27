#include "../include/TensorBLAS.h" 

void tc_cumpsgemm_symm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc)
{
    dim3 grida((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    copy_lower_to_upper<<<grida, block>>>(m, A, m);
    cumpsgemm::gemm(
            cumpsgemm_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m, n, m,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc,
            CUMPSGEMM_AUTO
            );
}