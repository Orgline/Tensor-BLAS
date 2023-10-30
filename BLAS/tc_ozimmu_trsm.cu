#include "../include/TensorBLAS.h" 

void tc_ozimmu_trsm_p2(cublasHandle_t handle, long int m, long int n, double* A, long int lda, double* B, long int ldb, long int nb)
{
    if(n <= nb)
    {
        //startTimer();
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDtrsm(handle,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            m, n, &sonedouble,
            A, lda,
            B, ldb
        );
        return;
    }
    
    tc_ozimmu_trsm_p2(handle, m, n/2, A, lda, B, ldb, nb);

    // cumpsgemm::gemm(
    //     ozimmu_handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_T,
    //     m, n/2, n/2,
    //     &snegone,
    //     B, ldb,
    //     A+n/2, lda,
    //     &sone,
    //     B+n/2*ldb, ldb,
    //     ozimmu_FP16TCEC
    //     );
    double sonedouble = 1.0, snegonedobule = -1.0;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n/2, n/2,
        &snegonedobule, B, CUDA_R_64F, m, A+n/2, CUDA_R_64F, n/2,
        &sonedouble, B+n/2*ldb, CUDA_R_64F, ldb, CUDA_R_64F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    tc_ozimmu_trsm_p2(handle, m, n/2, A+n/2*lda+n/2, lda, B+n/2*ldb, ldb, nb);
}
 
void tc_ozimmu_trsm_p3(cublasHandle_t handle, long int m, long int n, double* A, long int lda, double* B, long int ldb, long int nb)
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
            // printf("now nn=%d i = %d check ok\n", nn, i);
            tc_ozimmu_trsm_p2(handle, m, nn, A+offset+offset*lda, lda, B+offset*ldb, ldb, nb);
            // printf("check ok\n");
        }
        else
        {
            double sonedouble = 1.0, snegonedobule = -1.0;
            cublasDtrsm(handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                m, nn, &sonedouble,
                A+offset+offset*lda, lda,
                B+offset*ldb, ldb
            );
            
        }
        
        if(i != 0)
        {
            rest_n -=  nn;
            // cublasGemmEx(
            //         handle,
            //         CUBLAS_OP_N,
            //         CUBLAS_OP_T,
            //         m, rest_n, nn,
            //         &snegone,
            //         B+offset*ldb, ldb,
            //         A+offset+nn+offset*lda, lda,
            //         &sone,
            //         B+(offset+nn)*ldb, ldb,
            //         CUBLAS_GEMM_DEFAULT_TENSOR_OP
            //         );
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, rest_n, nn,
                        &snegone, B+offset*ldb, CUDA_R_64F, m, A+offset+nn+offset*lda, CUDA_R_64F, rest_n,
                        &sone, B+(offset+nn)*ldb, CUDA_R_64F, ldb, CUDA_R_64F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        

    }

}

void tc_ozimmu_trsm(cublasHandle_t handle, long int m, long int n, double* A, long int lda, double* B, long int ldb, long int nb)
{
    if(n%2||m%2) {
        double *A_, *B_;
        long int N = n, M = m, lda_, ldb_;
        n += n%2;
        m += m%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        cudaMalloc(&A_, sizeof(double)*n*n);
        cudaMalloc(&B_, sizeof(double)*m*n);
        printf("%ld, %ld\n", m, n);
        dim3 grid1((n+31)/32, (n+31)/32);
        dim3 block(32,32);
        setInitialValueDouble<<<grid1, block>>>(n, n ,A_, lda_, 0.0);
        dim3 grid2((m+31)/32, (n+31)/32);
        setInitialValueDouble<<<grid2, block>>>(m, n ,B_, ldb_, 0.0);

        dim3 grid3((N+31)/32, (N+31)/32);
        matrixCpyDouble<<<grid3, block>>>(N, N, A, lda, A_, lda_);
        dim3 grid4((M+31)/32, (N+31)/32);
        matrixCpyDouble<<<grid4, block>>>(M, N, B, ldb, B_, ldb_);

        tc_ozimmu_trsm_p3(handle, m, n, A_, lda_, B_, ldb_, nb);

        matrixCpyDouble<<<grid4, block>>>(M, N, B_, ldb_, B, ldb);
        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);

    }
    else {
        tc_ozimmu_trsm_p3(handle, m, n, A, lda, B, ldb, nb);
    }
}