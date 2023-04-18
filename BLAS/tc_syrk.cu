#include "../include/TensorBLAS.h" 

void tc_syrk_p2(cublasHandle_t handle, long int n, long int k, float alpha, __half* Ah, long int lda, float beta, float* C, long int ldc, long int nb)
{
    //printf("tc_syrk_p2\n");
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &alpha,
                                Ah, CUDA_R_16F, lda, nb,
                                Ah, CUDA_R_16F, lda, nb,
                                &beta, C, CUDA_R_32F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                   i*nb, i*nb, k, &alpha,
                                   Ah+i*nb, CUDA_R_16F, lda, 2*i*nb,
                                   Ah, CUDA_R_16F, lda, 2*i*nb,
                                   &beta, C+i*nb, CUDA_R_32F, ldc, 2*(i*nb+i*nb*lda),
                                   n/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

void tc_syrk(cublasHandle_t handle, long int n, long int k,  float alpha, float* A, long int lda, float beta, float* C, long int ldc, __half* Ah, long int nb)
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
    for(int i = length; i>=0; i--)
    {
        dim3 grid((n+31)/32, (k+31)/32);
        dim3 block(32,32);
        s2h<<<grid, block>>>(n, k, A, lda, Ah, lda);

        int nn = matSize[i];
        

        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        //printf("n = %ld, offset = %d\n", nn ,offset);   

        if(n % 8192 ==0 )
        {
            tc_syrk_p2(handle, nn, k, alpha, Ah+offset, lda, beta, C+offset+offset*ldc, ldc, nb);
        }
        else
        {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, nn, nn, k,
                    &alpha, Ah+offset, CUDA_R_16F, lda, Ah+offset, CUDA_R_16F, lda,
                    &beta, C+offset+offset*ldc, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if(i != 0)
        {
            rest_n -=  nn;
            //printf("rest_n = %d, nn = %d, offset = %d\n", rest_n, nn, offset);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rest_n, nn, k,
                    &alpha, Ah+offset+nn, CUDA_R_16F, lda, Ah+offset, CUDA_R_16F, lda,
                    &beta, C+offset+offset*ldc+nn, CUDA_R_32F, ldc, CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else
            return;
        
    }
    return;
}