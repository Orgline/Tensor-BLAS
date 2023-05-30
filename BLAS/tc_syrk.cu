#include "../include/TensorBLAS.h" 

bool syrk_python_flag = false;
int tc_syrk_wrapper(long int n, long int k, float* A, float* C, long int nb)
{
    syrk_python_flag =true;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*n*k);

    float alpha = sone;
    float beta = szero;

    // float *tmp;
    // cudaMalloc(&tmp, sizeof(float)*n*k);
    // dim3 grida((k+31)/32, (n+31)/32);
    // dim3 blocka(32,32);
    // transpose<<<grida, blocka>>>(k, n ,A, tmp);
    // printMatrixDeviceBlock("A.csv",n, k, A, n);
    // cudaFree(tmp);

    tc_syrk(cublas_handle, n, k, alpha, A, n, beta, C, n, hwork, nb);
    dim3 gridc((n+31)/32, (n+31)/32);
    dim3 blockc(32,32);
    copy_lower_to_upper<<<gridc, blockc>>>(n, C, n);

    cudaFree(hwork);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

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
    int offset;
    int rest_n = n;

    if(!syrk_python_flag)
    {
        constexpr auto block_size = 256;
	    constexpr auto smem_len = block_size * 16;
	    const auto grid_size = k;
        s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_size, block_size>>>(
					n, k,
					A, lda,
					Ah, lda
					);
    }
    else
    {
        dim3 grid((k+31)/32, (n+31)/32);
        dim3 block(32,32);
        s2hTranspose<<<grid, block>>>(k, n, A, Ah);
    }

    for(int i = length; i>=0; i--)
    {

        int nn = matSize[i];
        

        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        //printf("n = %ld, offset = %d\n", nn ,offset);   

        if(nn % 8192 ==0 )
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