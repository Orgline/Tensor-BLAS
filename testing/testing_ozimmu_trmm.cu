#include "../include/TensorBLAS.h"

long int m, n, nb;
double alpha;
bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    if(argc < 6)
    {
        printf("Needs m, n and nb, alpha, beta, check as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    alpha = atof(argv[4]);
    if (atoi(argv[5]) == 1)
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
    cudaMalloc(&A, sizeof(double)*m*m);

    double *B;
    cudaMalloc(&B, sizeof(double)*m*n);

    double *C;
    cudaMalloc(&C, sizeof(double)*m*n);

    generateUniformMatrixDouble(A, m, m);

    generateUniformMatrixDouble(B, m, n);

    dim3 gridc((m+31)/32, (n+31)/32);
    dim3 blockc(32,32);

    setInitialValueDouble<<<gridc, blockc>>>(m, n ,C, m, 0.0);

    dim3 grida((m+31)/32, (m+31)/32);
    clearTriDouble<<<grida, blockc>>>('u', m, m, A, m);

    // __half *hwork;
    // cudaMalloc(&hwork, sizeof(__half)*(m*n+m*m));
    
    // cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k,
    //             &alpha, hwork, CUDA_R_16F, n, hwork, CUDA_R_16F, n,
    //             &beta, C, CUDA_R_32F, n, CUDA_R_32F,
    //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                &alpha, A, CUDA_R_64F, m, A, CUDA_R_64F, m,
                &szero, C, CUDA_R_64F, m, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    startTimer();
    
    // s2h<<<grida, blockc>>>(m, m, A, m, hwork, m);
    // s2h<<<gridc, blockc>>>(m, n, B, m, hwork, m);
    

    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m,
                &alpha, A, CUDA_R_64F, m, A, CUDA_R_64F, m,
                &szero, C, CUDA_R_64F, m, CUDA_R_64F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    float ms = stopTimer();
    
    printf("tc_64gemm %dx%d takes %f ms, flops is %f\n", m, n, ms, 1.0*m*n*m/ms/1e9);

    //printf("snorm C = %f\n", snorm(n, n, C, n));
    setInitialValueDouble<<<gridc, blockc>>>(m, n ,C, m, 0.0);

    
    //setInitialValue<<<gridc, blockc>>>(n, k ,A, n, 0.5);
    //printf("snorm A = %f\n", snorm(n, k, A, n));

    
    startTimer();
    tc_ozimmu_trmm(cublas_handle, m, n, alpha, A, m, B, m, C, m, nb);
    ms = stopTimer();
    //printMatrixDeviceBlock("tC.csv", m, n, C, m);
    
    printf("tc_ozimmu_trmm %dx%d takes %f ms, flops is %f\n", m, n, ms, 1.0*m*n*m/ms/1e9);
    
    if(checkFlag)
    {
        double *tC;
        cudaMalloc(&tC, sizeof(double)*m*n);
        setInitialValueDouble<<<gridc, blockc>>>(m, n ,tC, m, 0.0);
        startTimer();

        cublasDtrmm(cublas_handle,
                    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    m, n, &alpha,
                    A, m, B, m, tC, m);
        ms = stopTimer();
    //printMatrixDeviceBlock("tC.csv", m, n, C, m);
    
        printf("tc_cublas_Dtrmm %dx%d takes %f ms, flops is %f\n", m, n, ms, 1.0*m*n*m/ms/1e9);
        // printMatrixDeviceBlock("A.csv", m, m, A, m);
        // printMatrixDeviceBlock("B.csv", m, n, B, m);
        // printMatrixDeviceBlock("C.csv", m, n, tC, m);
        //printf("snorm tC = %f\n", snorm(n, n, tC, n));
        //printMatrixDeviceBlock("C_a.csv", n, n, tC, n);
        
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                &sonedouble, C, m, &snegonedobule, tC, m,
                C, m);
        
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
        printf("snorm C = %f\n", snormDouble(m, n, C, m));
        printf("snorm tC = %f\n", snormDouble(m, n, tC, m));
        //cudaDeviceSynchronize();
        printf("Forward error is %.6e\n",snormDouble(m, n, C, m)/snormDouble(m, n, tC, m));
        cudaFree(tC);
    }
    cudaFree(C);
    // cudaFree(hwork);
    cudaFree(A);
    cudaFree(B);
    
    //}

}

