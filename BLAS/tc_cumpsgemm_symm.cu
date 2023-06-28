#include "../include/TensorBLAS.h" 

void tc_cumpsgemm_symm_p2(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc)
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
void tc_cumpsgemm_symm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc)
{
    if(n%2||m%2) {
        float *A_, *C_, *B_;
        long int N = n, M = m, lda_, ldb_, ldc_;
        n += n%2;
        m += m%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        ldc_ = ldc + ldc%2;
        cudaMalloc(&A_, sizeof(float)*m*m);
        cudaMalloc(&B_, sizeof(float)*m*n);
        cudaMalloc(&C_, sizeof(float)*m*n);
        printf("%ld, %ld\n", m, n);
        dim3 grid((m+31)/32, (n+31)/32);
        dim3 block(32,32);
        setInitialValue<<<grid, block>>>(m, m ,A_, lda_, 0.0);
        setInitialValue<<<grid, block>>>(m, n ,B_, ldb_, 0.0);
        setInitialValue<<<grid, block>>>(m, n ,C_, ldc_, 1.0);

        matrixCpy<<<grid, block>>>(M, M, A, lda, A_, lda_);
        matrixCpy<<<grid, block>>>(M, N, B, ldb, B_, ldb_);

        tc_cumpsgemm_symm_p2(cumpsgemm_handle, m, n, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_);

        matrixCpy<<<grid, block>>>(M, N, C_, ldc_, C, ldc);
        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_cumpsgemm_symm_p2(cumpsgemm_handle, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return;
}