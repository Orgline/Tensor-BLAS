#include "../include/TensorBLAS.h"

long int m, n, nb;
bool checkFlag = false;

int parseArguments(int argc,char *argv[])
{
    if(argc < 5)
    {
        printf("Needs m, n and nb, check as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    if (atoi(argv[4]) == 1)
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
    cudaMalloc(&A, sizeof(double)*n*n);
    double *B;
    cudaMalloc(&B, sizeof(double)*m*n);

    // __half *hwork;
    // cudaMalloc(&hwork, sizeof(__half)*(n/2*n/2+m/2*n));

    generateNormalMatrixDouble(A, n, n);
    generateUniformMatrixDouble(B, m, n);
    // dim3 gridb((m+31)/32, (n+31)/32);
    // dim3 blockb(32,32);
    // setInitialValue<<<gridb, blockb>>>(m, n ,B, m, 1.0);

    dim3 grid((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    setInitialValueDouble<<<grid, block>>>(n, n ,A, n, 0.1);
    clearTriDouble<<<grid, block>>>('u', n, n, A, n);
    //printMatrixDeviceBlock("A.csv", n, n, A, n);

    double *work;
    if(checkFlag)
    {
        cudaMalloc(&work, sizeof(double)*m*n);
        cudaMemcpy(work, B, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
    }
    
    startTimer();
    tc_ozimmu_trsm(cublas_handle, m, n, A, n, B, m, nb);
    float ms = stopTimer();

    printf("tc_ozimmu_trsm takes %f ms, flops is %f\n", ms, 1.0*m*n*n/ms/1e9);

    
    if(checkFlag)
    {
        startTimer();
        double sonedouble = 1.0, snegonedobule = -1.0;
        cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                m, n, &sonedouble,
                A, n,
                work, m
            );
        float ms = stopTimer();
        printf("Dtrsm takes %f ms, flops is %f\n", ms, 1.0*m*n*n/ms/1e9);
        // printMatrixDeviceBlock("B.csv", m, n, B, m);
        // printMatrixDeviceBlock("work.csv", m, n, work, m);

        cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                &sonedouble, B, m, &snegonedobule, work, m,
                work, m);
        //printMatrixDeviceBlock("work.csv", m, n, work, m);
        printf("Forward error ||X_tc-X_cublas||/||X_cublas|| is %.6e\n", snormDouble(m,n,work,m)/snormDouble(m,n,B,m));
    }
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }


}