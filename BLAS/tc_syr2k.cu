#include "../include/TensorBLAS.h" 

void tc_syr2k_p2(cublasHandle_t handle, long int n, long int k, float alpha, __half* Ah, long int lda, __half* Bh, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    //printf("tc_syrk_p2\n");
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &alpha,
                                Ah, CUDA_R_16F, lda, nb,
                                Bh, CUDA_R_16F, ldb, nb,
                                &beta, C, CUDA_R_32F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &alpha,
                                Bh, CUDA_R_16F, ldb, nb,
                                Ah, CUDA_R_16F, lda, nb,
                                &sone, C, CUDA_R_32F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                   i*nb, i*nb, k, &alpha,
                                   Ah+i*nb, CUDA_R_16F, lda, 2*i*nb,
                                   Bh, CUDA_R_16F, ldb, 2*i*nb,
                                   &beta, C+i*nb, CUDA_R_32F, ldc, 2*(i*nb+i*nb*lda),
                                   n/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                   i*nb, i*nb, k, &alpha,
                                   Bh+i*nb, CUDA_R_16F, ldb, 2*i*nb,
                                   Ah, CUDA_R_16F, lda, 2*i*nb,
                                   &sone, C+i*nb, CUDA_R_32F, ldc, 2*(i*nb+i*nb*lda),
                                   n/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

void tc_syr2k(cublasHandle_t handle, long int n, long int k,  float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, __half* hwork, long int nb)
{
    
    int length;
    int64_t* matSize = find_mat_size_syrk(n, &length);
    // for(int i = 0; i<=length; i++)
    // {
    //     printf("%ld ", matSize[i]);
    // }
    // printf("\n");
    int offset;
    int rest_n = n;

    __half *Ah = hwork;
    __half *Bh = hwork + n*k;

    dim3 grid((n+31)/32, (k+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n, k, A, lda, Ah, lda);
    s2h<<<grid, block>>>(n, k, B, ldb, Bh, ldb);

    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        

        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;

        if(nn % 8192 ==0 )
        {
            tc_syr2k_p2(handle, nn, k, alpha, Ah+offset, lda, Bh+offset, ldb, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, Ah+offset, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
                    &beta, C+offset+offset*ldc, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, Bh+offset, CUDA_R_16F, ldb, Ah+offset, CUDA_R_16F, lda,
                    &sone, C+offset+offset*ldc, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if(i != 0)
        {
            rest_n -=  nn;
            //printf("rest_n = %d, nn = %d, offset = %d\n", rest_n, nn, offset);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, Ah+offset+nn, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
                    &beta, C+offset+offset*ldc+nn, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, Bh+offset+nn, CUDA_R_16F, ldb, Ah+offset, CUDA_R_16F, lda,
                    &sone, C+offset+offset*ldc+nn, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else
            return;
        
    }
    return;
}