#include "../include/TensorBLAS.h" 


void tc_trmm_p2(cublasHandle_t handle, long int m, long int n, float alpha, __half* Ah, long int lda, __half* Bh, long int ldb, float beta, float* C, long int ldc, long int nb)
{
    //beginTimer();
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                nb, n, nb, &alpha,
                                Ah, CUDA_R_16F, lda, nb+nb*lda,
                                Bh, CUDA_R_16F, ldb, nb,
                                &beta, C, CUDA_R_32F, ldc, nb,
                                m/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //float ms = endTimer();
    //printf("batch gemm size %dx%dx%d takes %fms, rate is %f TFLOPs\n", nb, n, nb, ms, 2.0*nb*n*nb*m/nb/ms/1e9);
    for(long int i = 1; m / nb / i / 2 >= 1; i*=2)
    {
        //beginTimer();
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                   i*nb, n, i*nb, &alpha,
                                   Ah+i*nb, CUDA_R_16F, lda, 2*(i*nb+i*nb*lda),
                                   Bh, CUDA_R_16F, ldb, 2*i*nb,
                                   &sone, C+i*nb, CUDA_R_32F, ldc, 2*i*nb,
                                   m/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        //ms = endTimer();
        //printf("batch gemm size %dx%dx%d takes %fms, rate is %f TFLOPs\n", i*nb, n, i*nb, ms, 2.0*i*nb*n*i*nb*m/nb/i/2.0/ms/1e9);
    }
}

void tc_trmm_p3(cublasHandle_t handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, __half* hwork, long int nb)
{
    int length;
    int64_t* matSize = find_mat_size_syrk(m, &length);
    // for(int i = 0; i<=length; i++)
    // {
    //     printf("%ld ", matSize[i]);
    // }
    // printf("\n");
    int offset = 0;
    int rest_m = m;

    __half* Ah = hwork;
    __half* Bh = hwork + m*m;
    dim3 gridA((m+31)/32, (m+31)/32);
    dim3 block(32,32);
    s2h<<<gridA, block>>>(m, m, A, lda, Ah, lda);

    dim3 gridB((m+31)/32, (n+31)/32);
    s2h<<<gridB, block>>>(m, n, B, ldb, Bh, ldb);


    for(int i = length; i>=0; i--)
    {
        int mm = matSize[i];
        float beta;
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(i == length)
            beta = szero;
        else
            beta = sone;
        if (mm % 8192 == 0)
        {
            tc_trmm_p2(handle, mm, n, alpha, Ah+offset+offset*lda, lda, Bh+offset, ldb, beta, C+offset, ldc, nb);
        }
        else
        {
            //printf("offset = %d\n", offset);
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, mm,
                &alpha, Ah+offset+offset*lda, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
                &beta, C+offset, CUDA_R_32F, ldc, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if(i != 0)
        {
            rest_m -= mm;         
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, rest_m, n, mm,
                &alpha, Ah+offset+mm+offset*lda, CUDA_R_16F, lda, Bh+offset, CUDA_R_16F, ldb,
                &beta, C+offset+mm, CUDA_R_32F, ldc, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
}
void tc_trmm(cublasHandle_t handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, __half* hwork, long int nb)
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

        tc_trmm_p3(handle, m, n, alpha, A_, lda_, B_, ldb_, C_, ldc_, hwork, nb);

        matrixCpy<<<grid, block>>>(M, N, C_, ldc_, C, ldc);

        printf("check ok\n");
        cudaFree(A_);
        cudaFree(B_);
        cudaFree(C_);
    }
    else {
        tc_trmm_p3(handle, m, n, alpha, A, lda, B, ldb, C, ldc, hwork, nb);
    }
    return;
}