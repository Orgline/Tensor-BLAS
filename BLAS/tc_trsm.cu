#include "../include/TensorBLAS.h" 

void tc_rtrsm_p2(cublasHandle_t handle, long int m, long int n, float* A, long int lda, float* B, long int ldb, __half* hwork, long int nb)
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
    
    tc_rtrsm_p2(handle, m, n/2, A, lda, B, ldb, hwork, nb);

    __half *Ah = hwork;
    __half *Bh = hwork+n/2*n/2;

    dim3 grid((n/2+31)/32, (n/2+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n/2, n/2, A+n/2, lda, Ah, n/2);

    dim3 grid1((m+31)/32, (n/2+31)/32);
    dim3 block1(32,32);
    s2h<<<grid1, block1>>>(m, n/2, B, ldb, Bh, m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n/2, n/2,
        &snegone, Bh, CUDA_R_16F, m, Ah, CUDA_R_16F, n/2,
        &sone, B+n/2*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    tc_rtrsm_p2(handle, m, n/2, A+n/2*lda+n/2, lda, B+n/2*ldb, ldb, hwork, nb);
}
 
void tc_trsm(cublasHandle_t handle, long int m, long int n, float* A, long int lda, float* B, long int ldb, __half* hwork, long int nb)
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
            tc_rtrsm_p2(handle, m, nn, A+offset+offset*lda, lda, B+offset*ldb, ldb, hwork, nb);
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
            __half* Ah = hwork;
            
            dim3 grid((rest_n+31)/32, (nn+31)/32);
            dim3 block(32,32);
            s2h<<<grid, block>>>(rest_n, nn, A+offset+nn, lda, Ah, rest_n);
            
            __half* Bh = hwork + nn*rest_n;
            dim3 grid1((m+31)/32, (nn+31)/32);
            dim3 block1(32,32);
            s2h<<<grid1, block1>>>(m, nn, B+offset*ldb, ldb, Bh, m);

            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, rest_n, nn,
                        &snegone, Bh, CUDA_R_16F, m, Ah, CUDA_R_16F, rest_n,
                        &sone, B+(offset+nn)*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );

        }
        
    


    }

}
