#include "../include/TensorBLAS.h"
#include <string>

cudaEvent_t begin, end;
void beginTimer()
{
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float endTimer()
{
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

cudaEvent_t start, stop;
void startTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float stopTimer()
{
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}



__global__ void frobenius_norm_kernel(int64_t m, int64_t n, float *A, int64_t lda, double *norm) {
    int64_t idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    for (int64_t i = idx_x; i < m; i += blockDim.x * gridDim.x) {
        for (int64_t j = idx_y; j < n; j += blockDim.y * gridDim.y) {
            float value = A[i + j * lda];
            double value_squared = double(value) * (value);
            atomicAdd(norm, value_squared);
        }
    }
}
float snorm(long int m, long int n, float *d_A, long int lda) 
{
    const long int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    double *d_norm;
    cudaMalloc(&d_norm, sizeof(double));
    cudaMemset(d_norm, 0, sizeof(double));

    frobenius_norm_kernel<<<gridDim, blockDim>>>(m, n, d_A, lda, d_norm);

    double norm;
    cudaMemcpy(&norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost);
    norm = sqrtf(norm);

    cudaFree(d_norm);

    return float(norm);
}

__global__
void transpose(int m, int n, float* dA, float *tmpA){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i<m && j<n) {
        tmpA[i+j*m] = dA[i+j*m];
    }

    __syncthreads();

    if (i<m && j<n) {
        dA[j+i*n] = tmpA[i+j*m];
    }
}


__global__
void s2h(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

__global__
void s2hTranspose(long int m, long int n, float *as, __half *ah)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[j + i*n] = __float2half(as[i + j*m]);
	}
}

__global__ 
void copy_lower_to_upper(long int n, float *A, long int lda)
{
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < n && j < n) {
		if(i < j)
            A[i+j*lda] = A[j+i*lda];
	}
}

void generateNormalMatrix(float *dA,long int m,long int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m*n, 0, 1);
}

void generateUniformMatrix(float *dA,long int m,long int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen,dA,long(m*n));
}

// float snorm(long int m,long int n,float* dA)
// {
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     float sn;
//     int incx = 1;
//     cublasNrm2Ex_64(handle, int64_t(int64_t(m)*int64_t(n)), dA, CUDA_R_32F, incx, &sn, CUDA_R_32F, CUDA_R_32F);
//     cublasDestroy(handle);
//     return sn;
// }


__global__
void setEye( long int m, long int n, float *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] = 1;
		else
			a[i+j*lda] = 0;
	}
}


__global__
void setEyePlus( long int m, long int n, float *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] += 1.0;
	}

}

__global__
void setInitialValue( long int m, long int n, float *a, long int lda, float val)
{
        long int i = threadIdx.x + blockDim.x * blockIdx.x;
        long int j = threadIdx.y + blockDim.y * blockIdx.y;
        if (i < m && j < n) {
                a[i+j*lda] = val;
        }
}

__global__
void clearTri(char uplo, long int m, long int n, float *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0;
		}
	}
}


void print_env() {
    cudaDeviceProp prop;
    int cudaversion;
    int driverversion;

    cudaGetDeviceProperties(&prop, 0);
    int mpcount, s2dratio;
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&s2dratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&mpcount, cudaDevAttrMultiProcessorCount, 0);
    cudaRuntimeGetVersion(&cudaversion);
    cudaDriverGetVersion(&driverversion);

    std::cout << "=== Device information ===" << std::endl;
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "CUDA Runtime Version: " << cudaversion << std::endl;
    std::cout << "CUDA Driver Version: " << driverversion << std::endl;
    std::cout << "NVCC Version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
    std::cout << "GMem " << prop.totalGlobalMem << std::endl;
    std::cout << "SMem per block " << prop.sharedMemPerBlock << std::endl;
    std::cout << "SMem per MP " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Regs per block " << prop.regsPerBlock << std::endl;
    std::cout << "Clock rate " << prop.clockRate << std::endl;
    std::cout << "L2 $ size " << prop.l2CacheSize << std::endl;
    std::cout << "# MP " << mpcount << std::endl;
    std::cout << "single-double perf ratio " << s2dratio << std::endl;
//    std::cout << "__CUAD_ARCH__ " << __CUDA_ARCH__ << std::endl;
    std::cout << "=== END Deivce Information ===\n" << std::endl;
}

size_t free_mem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}