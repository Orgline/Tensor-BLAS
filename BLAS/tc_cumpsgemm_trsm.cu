#include "../include/TensorBLAS.h" 

void tc_rtrsm_p2(cublasHandle_t handle, cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float* A, long int lda, float* B, long int ldb, long int nb)
{
    if(n <= nb)
    {
        //startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, lda,
            B, ldb
        );
        return;
    }
    
    tc_rtrsm_p2(handle, cumpsgemm_handle, m, n/2, A, lda, B, ldb, nb);

    // printMatrixDeviceBlock("B0.csv", m, n, B, ldb);
    
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n/2, n/2,
    // &snegone, B, CUDA_R_32F, ldb, A+n/2, CUDA_R_32F, lda,
    // &sone, B+n/2*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
    // CUBLAS_GEMM_DEFAULT);
    // printMatrixDeviceBlock("A1.csv", n, n, A, lda);
    // printMatrixDeviceBlock("B1.csv", m, n, B, ldb);
    
    cumpsgemm::gemm(
        cumpsgemm_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        m, n/2, n/2,
        &snegone,
        B, ldb,
        A+n/2, lda,
        &sone,
        B+n/2*ldb, ldb,
        CUMPSGEMM_AUTO
        );
    // printMatrixDeviceBlock("A2.csv", n, n, A, lda);
    // printMatrixDeviceBlock("B2.csv", m, n, B, ldb);

    // float *A_, *B_;
    // cudaMalloc(&A_, sizeof(float)*n/2*n/2);
    // dim3 grid((n/2+31)/32, (n/2+31)/32);
    // dim3 block(32,32);
    // matrixCpy<<<grid, block>>>(n/2, n/2, A+n/2, lda, A_, n/2);

    // cudaMalloc(&B_, sizeof(float)*m*n/2);
    // dim3 grid1((m+31)/32, (n/2+31)/32);
    // dim3 block1(32,32);
    // matrixCpy<<<grid1, block1>>>(m, n/2, B, ldb, B_, m);

    // printf("what happen here %d %d\n", m, n);
    // cumpsgemm::gemm(
    //     cumpsgemm_handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_T,
    //     m, n/2, n/2,
    //     &snegone,
    //     B_, m,
    //     A_, n/2,
    //     &sone,
    //     B+n/2*ldb, ldb,
    //     CUMPSGEMM_AUTO
    //     );
    // printMatrixDeviceBlock("A_.csv", n/2, n/2, A_, n/2);
    // printMatrixDeviceBlock("B_.csv", m, n/2, B_, m);
    // printMatrixDeviceBlock("B.csv", m, n, B, m);

    // cudaFree(A_);
    // cudaFree(B_);
    tc_rtrsm_p2(handle, cumpsgemm_handle, m, n/2, A+n/2*lda+n/2, lda, B+n/2*ldb, ldb, nb);
}
 
void tc_cumpsgemm_trsm(cublasHandle_t handle, cumpsgemm::handle_t cumpsgemm_handle,long int m, long int n, float* A, long int lda, float* B, long int ldb, long int nb)
{
    int length;
    int64_t* matSize = find_mat_size_trsm(n, &length);
    long int offset;
    long int rest_n = n;

    for(int i = length; i>=0; i--)
    {
        int64_t nn = matSize[i];
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(nn % 2048 == 0)
        {
         printf("now nn=%d i = %d check ok\n", nn, i);
            tc_rtrsm_p2(handle, cumpsgemm_handle, m, nn, A+offset+offset*lda, lda, B+offset*ldb, ldb, nb);
                    printf("check ok\n");
        }
        else
        {
            
            cublasStrsm(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                m, nn, &sone,
                A+offset+offset*lda, lda,
                B+offset*ldb, ldb
            );
            
        }
        
        if(i != 0)
        {
            rest_n -=  nn;
            // float *A_, *B_;
            // cudaMalloc(&A_, sizeof(float)*rest_n*nn);
            // dim3 grid((rest_n+31)/32, (nn+31)/32);
            // dim3 block(32,32);
            // matrixCpy<<<grid, block>>>(rest_n, nn, A+offset+nn+offset*lda, lda, A_, rest_n);
            
            // cudaMalloc(&B_, sizeof(float)*m*nn);
            // dim3 grid1((m+31)/32, (nn+31)/32);
            // dim3 block1(32,32);
            // matrixCpy<<<grid1, block1>>>(m, nn, B+offset*ldb, ldb, B_, m);

            cumpsgemm::gemm(
                    cumpsgemm_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    m, rest_n, nn,
                    &snegone,
                    B+offset*ldb, ldb,
                    A+offset+nn+offset*lda, lda,
                    &sone,
                    B+(offset+nn)*ldb, ldb,
                    CUMPSGEMM_AUTO
                    );
            cudaFree(A_);
            cudaFree(B_);
        }
        

    }

}
