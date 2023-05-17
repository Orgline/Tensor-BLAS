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

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

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
    for(int i =0; i < 5; i++){
    
    {
        startTimer();
        dim3 gridA((m+31)/32, (k+31)/32);
        dim3 gridB((k+31)/32, (n+31)/32);
        dim3 block(32,32);
        s2h<<<gridA, block>>>(m, k, A, m, hA, m);
        s2h<<<gridB, block>>>(k, n, B, k, hB, k);
        
       cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                &sone, hA, CUDA_R_16F, m, hB, CUDA_R_16F, n,
                &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
       float ms = stopTimer();
        
        printf("tc_gemm input: FP16, output: FP32, accumulate: FP32, %dx%dx%d takes %f ms, flops is %f\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    }

    {
        startTimer();
        

       cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                &sone, hA, CUDA_R_16F, m, hB, CUDA_R_16F,n,
                &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        float ms = stopTimer();
        
        printf("tc_gemm input: FP16, output: FP32, accumulate: FP32, without converting %dx%dx%d takes %f ms, flops is %f\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    }


    {
        startTimer();

       cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                &sone, A, CUDA_R_32F, m, B, CUDA_R_32F, n,
                &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        float ms = stopTimer();
        
        printf("tc_gemm input: FP32, output: FP32, accumulate: FP32_FAST16, %dx%dx%d takes %f ms, flops is %f\n\n\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    }

    // {
    //     startTimer();

    //    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
    //             &sone, A, CUDA_R_32F, m, B, CUDA_R_32F, k,
    //             &szero, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
    //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //     float ms = stopTimer();
        
    //     printf("tc_gemm input: FP32, output: FP32, accumulate: FP32, %dx%dx%d takes %f ms, flops is %f\n\n", m, n,k, ms, 2.0*m*n*k/ms/1e9);
    // }
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

