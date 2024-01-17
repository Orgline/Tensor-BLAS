#include "../include/TensorBLAS.h"

long int m, n, k;

int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Needs m, n and k as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    return 0;
}
__global__
void sethalfInitialValue( long int m, long int n, __half *a, long int lda, float val)
{
        long int i = threadIdx.x + blockDim.x * blockIdx.x;
        long int j = threadIdx.y + blockDim.y * blockIdx.y;
        if (i < m && j < n) {
                a[i+j*lda] = val;
        }
}
int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    bool outer_product = false;
    if(m == n){
        outer_product = true;
    }
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
	cumpsgemm::handle_t cumpsgemm_handle;
	cumpsgemm::create(cumpsgemm_handle);

    float *A;
    cudaMalloc(&A, sizeof(float)*m*k);

    float *B;
    cudaMalloc(&B, sizeof(float)*k*n);

    float *C;
    cudaMalloc(&C, sizeof(float)*m*n);

    //setInitialValue<<<gridc, blockc>>>(n, n ,C, n, 1.0);
    __half *hA;
    cudaMalloc(&hA, sizeof(__half)*m*k);
    __half *hB;
    cudaMalloc(&hB, sizeof(__half)*k*n);
     __half *hC;
    cudaMalloc(&hC, sizeof(__half)*m*n);
    __half *tmpC;
    cudaMalloc(&tmpC, sizeof(__half)*m*n);
    //warm up
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                &sone, hA, CUDA_R_16F, m, hB, CUDA_R_16F, n,
                &szero, C, CUDA_R_32F, m, CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        // Synchronize the device and check for kernel execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        outer_product = true;
        int tmpK = k;
    for(int i =0; i < 5; i++){
    {

        k = tmpK;
    //     if(outer_product&&k>=16384){
    //         int offset = 0;
    //         startTimer();
    //         dim3 gridA((m+31)/32, (k+31)/32);
    //         dim3 gridB((k+31)/32, (n+31)/32);
    //         dim3 block(32,32);

    //         constexpr auto block_size = 256;
    //         constexpr auto smem_len = block_size * 16;
    //         const auto grid_sizeA = k;
    //         const auto grid_sizeB = n;
    //         s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_sizeA, block_size>>>(
    //                     m, k,
    //                     A, m,
    //                     hA, m
    //                     );
    //         s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_sizeB, block_size>>>(
    //             k, n,
    //             B, k,
    //             hB, k
    //             );
    //    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
    //             &sone, hA, CUDA_R_16F, m, hB, CUDA_R_16F, n,
    //             &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
    //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //         // while(k >= 16384){
    //         //     k -= 16384;
    //         //     cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 16384,
    //         //     &sone, hA+offset*m, CUDA_R_16F, m, hB+offset*n, CUDA_R_16F, n,
    //         //     &sone, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
    //         //     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //         //     offset += 16384;

    //         // }
    //         // if(k!=0){
    //         //     cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
    //         //     &sone, hA+offset*m, CUDA_R_16F, m, hB+offset*n, CUDA_R_16F, n,
    //         //     &sone, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
    //         //     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //         // }
  
    //         float ms = stopTimer();
    //         if(i == 0)
    //         printf("16384tc_gemm and s2h_swpipe, %dx%dx%d takes %f ms, flops is %f\n", m, n,tmpK, ms, 2.0*m*n*tmpK/ms/1e9);
    //         dim3 gridC((m+31)/32, (n+31)/32);
    //         // sethalfInitialValue<<<gridC, block>>>(m, n, hC, m, 0);
    //         setInitialValue<<<gridC, block>>>(m, n, C, m, 0);
    //         continue;
    //     }
    }
    {
        startTimer();
        dim3 gridA((m+31)/32, (k+31)/32);
        dim3 gridB((k+31)/32, (n+31)/32);
        dim3 block(32,32);
        // s2h<<<gridA, block>>>(m, k, A, m, hA, m);
        // s2h<<<gridB, block>>>(k, n, B, k, hB, k);
        constexpr auto block_size = 256;
	    constexpr auto smem_len = block_size * 16;
	    const auto grid_sizeA = k;
        const auto grid_sizeB = n;
        s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_sizeA, block_size>>>(
					m, k,
					A, m,
					hA, m
					);
        s2h_swpipe<std::uint64_t, block_size, smem_len><<<grid_sizeB, block_size>>>(
            k, n,
            B, k,
            hB, k
            );
    //    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
    //             &sone, hA, CUDA_R_16F, m, hB, CUDA_R_16F, n,
    //             &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
    //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        int offset = 0;
            while(k >= 16384){
                k -= 16384;
                cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, 16384,
                &sone, hA+offset*m, CUDA_R_16F, m, hB+offset*n, CUDA_R_16F, n,
                &sone, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                offset += 16384;

            }
            if(k!=0){
                cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                &sone, hA+offset*m, CUDA_R_16F, m, hB+offset*n, CUDA_R_16F, n,
                &sone, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
       float ms = stopTimer();
       k = tmpK;
        if(i == 4)
        printf("tc_gemm and s2h_swpipe, %dx%dx%d takes %f ms, flops is %f\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
         dim3 gridC((m+31)/32, (n+31)/32);
                setInitialValue<<<gridC, block>>>(m, n, C, m, 0);
    }

    {
        startTimer();
        cumpsgemm::gemm(
        cumpsgemm_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        m, n, k,
        &sone,
        A, m,
        B, n,
        &szero,
        C, m,
        CUMPSGEMM_FP16TCEC
        );

        float ms = stopTimer();
        if(i == 4)
        printf("cumpsgemm, %dx%dx%d takes %f ms, flops is %f\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    }
    {
        startTimer();

        // cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
        //         &sone, A, CUDA_R_32F, m, B, CUDA_R_32F, k,
        //         &szero, C, CUDA_R_32F, m, CUDA_R_32F,
        //         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, k,
                    &sone,
                    A, m,
                    B, n,
                    &szero,
                    C, m);
        float ms = stopTimer();
        if(i == 4)
        printf("Sgemm, %dx%dx%d takes %f ms, flops is %f\n\n\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    }

     cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        // Synchronize the device and check for kernel execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    
    cudaFree(C);
    cudaFree(B);
    cudaFree(hB);
    cudaFree(hA);
    cudaFree(A);
    
    //}

}

