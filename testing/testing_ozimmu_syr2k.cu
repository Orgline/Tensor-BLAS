#include "../include/TensorBLAS.h"

long int n, k, nb;
double alpha, beta;
bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    if(argc < 7)
    {
        printf("Needs m, n and nb, alpha, beta, check as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    alpha = atof(argv[4]);
    beta= atof(argv[5]);
    if (atoi(argv[6]) == 1)
        checkFlag = true;
    else
        checkFlag = false;
    return 0;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
	// cumpsgemm::handle_t cumpsgemm_handle;
	// cumpsgemm::create(cumpsgemm_handle);

    double *A;
    cudaMalloc(&A, sizeof(double)*n*k);

    double *B;
    cudaMalloc(&B, sizeof(double)*n*k);

    double *C;
    cudaMalloc(&C, sizeof(double)*n*n);

    dim3 gridc((n+31)/32, (n+31)/32);
    dim3 blockc(32,32);


    setInitialValueDouble<<<gridc, blockc>>>(n, n ,C, n, 1.0);
    // __half *hwork;
    // cudaMalloc(&hwork, sizeof(__half)*n*k*2);
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
                &alpha, A, CUDA_R_64F, n, A, CUDA_R_64F, n,
                &beta, C, CUDA_R_64F, n, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    startTimer();
    // dim3 grid((n+31)/32, (k+31)/32);
    // dim3 block(32,32);
    // s2h<<<grid, block>>>(n, k, A, n, hwork, n);
    // s2h<<<grid, block>>>(n, k, A, n, hwork, n);
    
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
                &alpha, A, CUDA_R_64F, n, A, CUDA_R_64F, n,
                &beta, C, CUDA_R_64F, n, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
                &alpha, A, CUDA_R_64F, n, A, CUDA_R_64F, n,
                &beta, C, CUDA_R_64F, n, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    float ms = stopTimer();
    
    printf("Two tc_gemm %dx%d takes %f ms, flops is %f\n", n, k,ms, 2.0*n*n*k/ms/1e9);

    //printf("snorm C = %f\n", snorm(n, n, C, n));
    setInitialValueDouble<<<gridc, blockc>>>(n, n ,C, n, 1.0);

    generateUniformMatrixDouble(A, n, k);
    generateNormalMatrixDouble(B, n, k);
    //setInitialValue<<<gridc, blockc>>>(n, k ,A, n, 0.5);
    //printf("snorm A = %f\n", snorm(n, k, A, n));
    

    
    startTimer();
    tc_ozimmu_syr2k(cublas_handle, n, k, alpha, A, n, B, n, beta, C, n, nb);
    ms = stopTimer();
    
    printf("tc_ozimmu_syr2k %dx%d takes %f ms, flops is %f\n", n, k, ms, 2.0*n*n*k/ms/1e9);
    
    copy_lower_to_upperDouble<<<gridc, blockc>>>(n, C, n);
    //printf("snorm C = %f\n", snorm(n, n, C, n));
    //printMatrixDeviceBlock("C.csv", n, n, C, n);
    if(checkFlag)
    {
        double *tC;
        cudaMalloc(&tC, sizeof(double)*n*n);
        setInitialValueDouble<<<gridc, blockc>>>(n, n ,tC, n, 1.0);

       

        
        startTimer();
        cublasDsyr2k(cublas_handle,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            n, k,
            &alpha,
            A, n,
            B, n,
            &beta,
            tC, n
        );
        ms = stopTimer();
        printf("Dsyr2k %dx%d takes %f ms, flops is %f\n", n, k, ms, 2.0*n*n*k/ms/1e9);
    
        //printMatrixDeviceBlock("C_p.csv", n, n, tC, n);
        copy_lower_to_upperDouble<<<gridc, blockc>>>(n, tC, n);
        //printf("snorm tC = %f\n", snorm(n, n, tC, n));
        //printMatrixDeviceBlock("C_a.csv", n, n, tC, n);
    double sonedouble = 1.0, snegonedobule = -1.0; 

        cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                &sonedouble, C, n, &snegonedobule, tC, n,
                C, n);
        
        //sSubstract<<<gridc, blockc>>>(n, n, C, n, tC, n);
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
        printf("free memory is %d GB\n", free_mem()/1024/1024/1024);
        // printf("snorm C = %f\n", snorm(n, n, C, n));
        // printf("snorm tC = %f\n", snorm(n, n, tC, n));
        //cudaDeviceSynchronize();
        printf("Forward error is %.6e\n",snormDouble(n, n, C, n)/snormDouble(n, n, tC, n));
        cudaFree(tC);
    }
    cudaFree(C);
    // cudaFree(hwork);
    cudaFree(A);
    cudaFree(B);
    
    //}

}

