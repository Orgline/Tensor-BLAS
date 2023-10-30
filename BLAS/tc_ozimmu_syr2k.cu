#include "../include/TensorBLAS.h" 

void tc_ozimmu_syr2k_p2(cublasHandle_t handle, long int n, long int k, double alpha, double* A, long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb)
{
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &alpha,
                                A, CUDA_R_64F, lda, nb,
                                B, CUDA_R_64F, ldb, nb,
                                &beta, C, CUDA_R_64F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &alpha,
                                B, CUDA_R_64F, ldb, nb,
                                A, CUDA_R_64F, lda, nb,
                                &sone, C, CUDA_R_64F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                   i*nb, i*nb, k, &alpha,
                                   A+i*nb, CUDA_R_64F, lda, 2*i*nb,
                                   B, CUDA_R_64F, ldb, 2*i*nb,
                                   &beta, C+i*nb, CUDA_R_64F, ldc, 2*(i*nb+i*nb*lda),
                                   n/nb/i/2, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                   i*nb, i*nb, k, &alpha,
                                   B+i*nb, CUDA_R_64F, ldb, 2*i*nb,
                                   A, CUDA_R_64F, lda, 2*i*nb,
                                   &sone, C+i*nb, CUDA_R_64F, ldc, 2*(i*nb+i*nb*lda),
                                   n/nb/i/2, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

void tc_ozimmu_syr2k_p3(cublasHandle_t handle, long int n, long int k,  double alpha, double* A, long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb)
{
    
    int length;
    int64_t* matSize = find_mat_size_syrk(n, &length);
    int offset;
    int rest_n = n;

    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;

        if(nn % 8192 ==0 )
        {
            tc_ozimmu_syr2k_p2(handle, nn, k, alpha, A+offset, lda, B+offset, ldb, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, A+offset, CUDA_R_64F, lda, B+offset, CUDA_R_64F, ldb,
                    &beta, C+offset+offset*ldc, CUDA_R_64F, ldc, CUDA_R_64F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, B+offset, CUDA_R_64F, ldb, A+offset, CUDA_R_64F, lda,
                    &sone, C+offset+offset*ldc, CUDA_R_64F, ldc, CUDA_R_64F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if(i != 0)
        {
            rest_n -=  nn;
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, A+offset+nn, CUDA_R_64F, lda, B+offset, CUDA_R_64F, ldb,
                    &beta, C+offset+offset*ldc+nn, CUDA_R_64F, ldc, CUDA_R_64F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, B+offset+nn, CUDA_R_64F, ldb, A+offset, CUDA_R_64F, lda,
                    &sone, C+offset+offset*ldc+nn, CUDA_R_64F, ldc, CUDA_R_64F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else
            return;
        
    }
    return;
}
void tc_ozimmu_syr2k(cublasHandle_t handle, long int n, long int k,  double alpha, double* A, long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb)
{
    if(n%2||k%2) {
        double *A_, *C_, *B_;
        long int N = n, K = k, lda_, ldb_, ldc_;
        n += n%2;
        k += k%2;
        lda_ = lda + lda%2;
        ldb_ = ldb + ldb%2;
        ldc_ = ldc + ldc%2;
        cudaMalloc(&A_, sizeof(double)*n*k);
        cudaMalloc(&B_, sizeof(double)*n*k);
        cudaMalloc(&C_, sizeof(double)*n*n);
        printf("%ld, %ld\n", n, k);
        dim3 grid1((n+31)/32, (k+31)/32);
        dim3 block(32,32);
        setInitialValueDouble<<<grid1, block>>>(n, k ,A_, lda_, 0.0);
        setInitialValueDouble<<<grid1, block>>>(n, k ,B_, ldb_, 0.0);
        dim3 grid2((n+31)/32, (n+31)/32);
        setInitialValueDouble<<<grid2, block>>>(n, n ,C_, ldc_, 1.0);
        dim3 grid3((N+31)/32, (K+31)/32);
        matrixCpyDouble<<<grid3, block>>>(N, K, A, lda, A_, lda_);
        matrixCpyDouble<<<grid3, block>>>(N, K, B, ldb, B_, ldb_);


        tc_ozimmu_syr2k_p3(handle, n, k, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_, nb);
        dim3 grid4((N+31)/32, (N+31)/32);
        matrixCpyDouble<<<grid4, block>>>(N, N, C_, ldc_, C, ldc);

        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_ozimmu_syr2k_p3(handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nb);
    }
}